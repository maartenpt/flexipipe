"""
Models module for FlexiPipe.
"""
import torch
from torch import nn
from transformers import AutoModel, Trainer, TrainingArguments

class BiaffineAttention(nn.Module):
    """Biaffine attention for dependency head prediction."""
    def __init__(self, hidden_size: int, arc_dim: int = 500):
        super().__init__()
        self.arc_dim = arc_dim
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_size, arc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(arc_dim, arc_dim)
        )
        self.dep_mlp = nn.Sequential(
            nn.Linear(hidden_size, arc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(arc_dim, arc_dim)
        )
        # Biaffine layer: for each (head, dep) pair, compute score
        # Use Bilinear layer: head @ W @ dep.T
        self.arc_biaffine = nn.Bilinear(arc_dim, arc_dim, 1, bias=True)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            arc_scores: [batch_size, seq_len, seq_len] - scores for head predictions
                arc_scores[i, j] = score for token j having head i
        """
        head_repr = self.head_mlp(hidden_states)  # [batch, seq, arc_dim]
        dep_repr = self.dep_mlp(hidden_states)     # [batch, seq, arc_dim]
        
        batch_size, seq_len, arc_dim = head_repr.shape
        
        # Safety check: truncate if sequence is too long
        if seq_len > 512:
            seq_len = 512
            head_repr = head_repr[:, :seq_len, :]
            dep_repr = dep_repr[:, :seq_len, :]
        
        # Memory-efficient biaffine computation using batched matrix multiplication
        # Instead of expand(), use broadcasting and batch operations
        # head_repr: [batch, seq, arc_dim], dep_repr: [batch, seq, arc_dim]
        # We want: [batch, seq, seq] where score[i,j] = biaffine(head[i], dep[j])
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 64  # Process 64 tokens at a time (much smaller to avoid memory issues)
        arc_scores_list = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            head_chunk = head_repr[:, i:end_i, :]  # [batch, chunk_i, arc_dim]
            chunk_i = end_i - i
            
            row_scores = []
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                dep_chunk = dep_repr[:, j:end_j, :]  # [batch, chunk_j, arc_dim]
                chunk_j = end_j - j
                
                # Compute scores for this chunk pair without expand
                # head_chunk: [batch, chunk_i, arc_dim]
                # dep_chunk: [batch, chunk_j, arc_dim]
                # We need: [batch, chunk_i, chunk_j]
                
                # Use repeat instead of expand (more memory efficient for small chunks)
                head_exp = head_chunk.unsqueeze(2).repeat(1, 1, chunk_j, 1)  # [batch, chunk_i, chunk_j, arc_dim]
                dep_exp = dep_chunk.unsqueeze(1).repeat(1, chunk_i, 1, 1)   # [batch, chunk_i, chunk_j, arc_dim]
                
                # Flatten for biaffine
                head_flat = head_exp.reshape(-1, arc_dim)
                dep_flat = dep_exp.reshape(-1, arc_dim)
                
                # Compute biaffine scores
                scores_flat = self.arc_biaffine(head_flat, dep_flat)  # [batch * chunk_i * chunk_j, 1]
                scores = scores_flat.reshape(batch_size, chunk_i, chunk_j)
                row_scores.append(scores)
            
            # Concatenate along j dimension
            if row_scores:
                row = torch.cat(row_scores, dim=2)  # [batch, chunk_i, seq_len]
                arc_scores_list.append(row)
        
        # Concatenate along i dimension
        if arc_scores_list:
            arc_scores = torch.cat(arc_scores_list, dim=1)  # [batch, seq_len, seq_len]
        else:
            arc_scores = torch.zeros(batch_size, seq_len, seq_len, device=head_repr.device, dtype=head_repr.dtype)
        
        return arc_scores



class MultiTaskFlexiPipeTagger(nn.Module):
    """Multi-task FlexiPipe tagger and parser with separate heads for UPOS, XPOS, FEATS, lemmatizer, and parsing."""
    
    def __init__(self, base_model_name: str, num_upos: int, num_xpos: int, num_feats: int, 
                 num_lemmas: int = 0, num_deprels: int = 0, num_norms: int = 0,
                 train_parser: bool = False, train_lemmatizer: bool = False, train_normalizer: bool = False,
                 num_orig_forms: int = 0, use_orig_form_for_parser: bool = False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.train_parser = train_parser
        self.train_lemmatizer = train_lemmatizer
        self.train_normalizer = train_normalizer
        self.num_upos = num_upos
        self.num_xpos = num_xpos
        self.num_feats = num_feats
        
        # Classification heads for tagging - use MLPs instead of simple Linear
        # This is crucial for SOTA performance
        mlp_hidden = hidden_size // 2  # Half the hidden size for intermediate layer
        
        self.upos_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_upos)
        )
        self.xpos_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_xpos)
        )
        self.feats_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_feats)
        )
        
        # Lemmatizer head (if training lemmatizer)
        # Context-aware: use UPOS/XPOS/FEATS embeddings + BERT embeddings
        if train_lemmatizer and num_lemmas > 0:
            # Embedding dimensions for categorical features
            upos_embed_dim = 32  # Small embedding for UPOS
            xpos_embed_dim = 64  # Larger embedding for XPOS (more specific)
            feats_embed_dim = 32  # Embedding for FEATS
            
            # Embedding layers for categorical features
            self.lemma_upos_embed = nn.Embedding(num_upos, upos_embed_dim)
            self.lemma_xpos_embed = nn.Embedding(num_xpos, xpos_embed_dim)
            self.lemma_feats_embed = nn.Embedding(num_feats, feats_embed_dim)
            
            # Combined input size: BERT hidden + UPOS + XPOS + FEATS embeddings
            combined_hidden = hidden_size + upos_embed_dim + xpos_embed_dim + feats_embed_dim
            
            self.lemma_head = nn.Sequential(
                nn.Linear(combined_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_lemmas)
            )
            
            # Store embedding dimensions for later use
            self.lemma_upos_embed_dim = upos_embed_dim
            self.lemma_xpos_embed_dim = xpos_embed_dim
            self.lemma_feats_embed_dim = feats_embed_dim
        else:
            self.lemma_head = None
            self.lemma_upos_embed = None
            self.lemma_xpos_embed = None
            self.lemma_feats_embed = None
        
        # Original form embedding for transpositional parsing (if enabled)
        self.use_orig_form_for_parser = use_orig_form_for_parser
        self.orig_form_embed = None
        orig_form_embed_dim = 64  # Embedding dimension for original forms
        if use_orig_form_for_parser and num_orig_forms > 0:
            self.orig_form_embed = nn.Embedding(num_orig_forms, orig_form_embed_dim)
            # Adjust hidden size for parser to account for orig_form embeddings
            parser_hidden_size = hidden_size + orig_form_embed_dim
        else:
            parser_hidden_size = hidden_size
        
        # Parsing heads (only if training parser)
        if train_parser and num_deprels > 0:
            # Use adjusted hidden size that includes orig_form embeddings if enabled
            self.biaffine = BiaffineAttention(parser_hidden_size, arc_dim=500)
            self.deprel_head = nn.Sequential(
                nn.Linear(parser_hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_deprels)
            )
        else:
            self.biaffine = None
            self.deprel_head = None
        
        # Normalizer head (if training normalizer)
        if train_normalizer and num_norms > 0:
            self.norm_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_norms)
            )
        else:
            self.norm_head = None
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels_upos=None, labels_xpos=None, 
                labels_feats=None, labels_lemma=None, labels_norm=None, labels_head=None, labels_deprel=None,
                orig_form_ids=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits_upos = self.upos_head(sequence_output)
        logits_xpos = self.xpos_head(sequence_output)
        logits_feats = self.feats_head(sequence_output)
        
        # Normalizer outputs
        logits_norm = None
        if self.train_normalizer and self.norm_head is not None:
            logits_norm = self.norm_head(sequence_output)  # [batch, seq, num_norms]
        
        # Lemmatizer outputs (context-aware: uses UPOS/XPOS/FEATS)
        logits_lemma = None
        if self.train_lemmatizer and self.lemma_head is not None:
            # Get predicted UPOS/XPOS/FEATS for context-aware lemmatization
            # Use predicted labels (argmax) during inference, or use provided labels during training
            batch_size, seq_len, _ = sequence_output.shape
            
            # Get UPOS/XPOS/FEATS predictions (or use provided labels if available)
            if labels_upos is not None:
                upos_ids = labels_upos  # Use ground truth during training
            else:
                upos_ids = torch.argmax(logits_upos, dim=-1)  # Use predictions during inference
            
            if labels_xpos is not None:
                xpos_ids = labels_xpos
            else:
                xpos_ids = torch.argmax(logits_xpos, dim=-1)
            
            if labels_feats is not None:
                feats_ids = labels_feats
            else:
                feats_ids = torch.argmax(logits_feats, dim=-1)
            
            # Embed UPOS/XPOS/FEATS
            upos_embeds = self.lemma_upos_embed(upos_ids)  # [batch, seq, upos_embed_dim]
            xpos_embeds = self.lemma_xpos_embed(xpos_ids)  # [batch, seq, xpos_embed_dim]
            feats_embeds = self.lemma_feats_embed(feats_ids)  # [batch, seq, feats_embed_dim]
            
            # Concatenate BERT embeddings with POS/FEATS embeddings
            combined_embeds = torch.cat([sequence_output, upos_embeds, xpos_embeds, feats_embeds], dim=-1)
            
            logits_lemma = self.lemma_head(combined_embeds)  # [batch, seq, num_lemmas]
        
        # Parsing outputs
        arc_scores = None
        logits_deprel = None
        if self.train_parser and self.biaffine is not None:
            # For transpositional parsing: concatenate original form embeddings with BERT embeddings
            parser_input = sequence_output
            if self.use_orig_form_for_parser and self.orig_form_embed is not None and orig_form_ids is not None:
                # Embed original forms: [batch, seq] -> [batch, seq, orig_form_embed_dim]
                orig_form_embeds = self.orig_form_embed(orig_form_ids)
                # Concatenate: [batch, seq, hidden_size] + [batch, seq, orig_form_embed_dim]
                parser_input = torch.cat([sequence_output, orig_form_embeds], dim=-1)
            
            arc_scores = self.biaffine(parser_input)  # [batch, seq, seq]
            # Deprel scores: for each possible head-child pair
            # We'll use a simpler approach: predict deprel for each token given its predicted head
            logits_deprel = self.deprel_head(parser_input)  # [batch, seq, num_deprels]
        
        loss = None
        if labels_upos is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Weight losses: UPOS is most important (2.0), XPOS (1.5), FEATS (1.0), Lemma (1.5)
            # This helps prioritize UPOS accuracy which is critical
            upos_loss = loss_fct(logits_upos.view(-1, logits_upos.size(-1)), labels_upos.view(-1))
            loss = 2.0 * upos_loss  # UPOS gets double weight
            
            if labels_xpos is not None:
                xpos_loss = loss_fct(logits_xpos.view(-1, logits_xpos.size(-1)), labels_xpos.view(-1))
                loss += 1.5 * xpos_loss  # XPOS gets 1.5x weight
            
            if labels_feats is not None:
                feats_loss = loss_fct(logits_feats.view(-1, logits_feats.size(-1)), labels_feats.view(-1))
                loss += 1.0 * feats_loss  # FEATS gets standard weight
            
            # Lemma loss
            if self.train_lemmatizer and labels_lemma is not None and logits_lemma is not None:
                lemma_loss = loss_fct(logits_lemma.view(-1, logits_lemma.size(-1)), labels_lemma.view(-1))
                loss += 1.5 * lemma_loss  # Lemma gets 1.5x weight (similar to XPOS)
            
            # Normalizer loss
            if self.train_normalizer and labels_norm is not None and logits_norm is not None:
                norm_loss = loss_fct(logits_norm.view(-1, logits_norm.size(-1)), labels_norm.view(-1))
                loss += 1.0 * norm_loss  # Normalizer gets standard weight
            
            # Parsing loss
            if self.train_parser and labels_head is not None and arc_scores is not None:
                # Arc loss: cross-entropy over heads (each token should have one head)
                batch_size, seq_len, _ = arc_scores.shape
                # Mask invalid positions (padding, special tokens)
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
                    mask = mask & mask.transpose(1, 2)  # Both dimensions must be valid
                    arc_scores = arc_scores.masked_fill(~mask.bool(), float('-inf'))
                
                # Arc loss: negative log-likelihood of correct head
                arc_loss = nn.CrossEntropyLoss(ignore_index=-100)
                loss += arc_loss(arc_scores.view(-1, seq_len), labels_head.view(-1))
                
                # Deprel loss: only for tokens with valid heads
                if labels_deprel is not None and logits_deprel is not None:
                    deprel_loss = nn.CrossEntropyLoss(ignore_index=-100)
                    loss += deprel_loss(logits_deprel.view(-1, logits_deprel.size(-1)), labels_deprel.view(-1))
        
            return {
                'loss': loss,
                'logits_upos': logits_upos,
                'logits_xpos': logits_xpos,
                'logits_feats': logits_feats,
                'logits_lemma': logits_lemma,
                'logits_norm': logits_norm,
                'arc_scores': arc_scores,
                'logits_deprel': logits_deprel,
            }
        
        # Inference mode (no labels): return logits without loss
        return {
            'logits_upos': logits_upos,
            'logits_xpos': logits_xpos,
            'logits_feats': logits_feats,
            'logits_lemma': logits_lemma,
            'logits_norm': logits_norm,
            'arc_scores': arc_scores,
            'logits_deprel': logits_deprel,
        }



