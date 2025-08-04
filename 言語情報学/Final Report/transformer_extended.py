import torch
import torch.nn as nn
import math
from decimal import Decimal, getcontext

# ========== Configurations =========
# Data settings
VOCAB_SIZE = 13  # 10 digits (0-9) + 3 special tokens (PAD, BOS, EOS)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
DIGIT_OFFSET = 3  # Map digits 0-9 to 3-12

# Model settings
D_MODEL = 32     # Embedding dimension
N_HEADS = 2       # Number of attention heads
N_LAYERS = 1      # Number of Transformer layers
DIM_FFN = 64     # Feedforward layer dimension
DROPOUT = 0.1     # Dropout rate
MAX_SEQ_LEN = 20  # Maximum sequence length

# Training settings
BATCH_SIZE = 32
BATCHES_PER_EPOCH = 4
LEARNING_RATE = 5e-4
N_EPOCHS = 100
SEQ_LENGTH = 10

# Other settings
PI_DIGITS = 1000  # Digits of π to compute
E_DIGITS = 1000   # Digits of e to compute
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Compute π and e =========
print("Calculating π and e...")
getcontext().prec = max(PI_DIGITS, E_DIGITS) + 100

# Compute π using Bailey–Borwein–Plouffe formula
PI = str(sum(Decimal(16)**(-k) * (
    Decimal(4)/(8*k+1) - Decimal(2)/(8*k+4) -
    Decimal(1)/(8*k+5) - Decimal(1)/(8*k+6)
) for k in range(PI_DIGITS//2)))[2:PI_DIGITS+2]

# Compute e using Taylor series
E = str(sum(Decimal(1)/math.factorial(k) for k in range(200)))[2:E_DIGITS+2]

print(f"Generated {len(PI)} digits of π and {len(E)} digits of e")

# ========== Model Definition ==========
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, D_MODEL) * math.sqrt(D_MODEL))

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            num_encoder_layers=N_LAYERS,
            num_decoder_layers=N_LAYERS,
            dim_feedforward=DIM_FFN,
            dropout=DROPOUT,
            batch_first=True
        )

        self.output_layer = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.d_model = D_MODEL

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # Target mask to prevent looking ahead
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

        # Embedding + positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, :src.size(1)]

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1)]

        # Transformer with explicit masks
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.output_layer(output)
    
class TinySharedTransformer(nn.Module):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 d_model=D_MODEL,
                 n_heads=N_HEADS,
                 dropout=DROPOUT,
                 max_len=MAX_SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos     = nn.Parameter(
            torch.randn(1, max_len, d_model) * math.sqrt(d_model)
        )

        # ---------- 1) 先建 Encoder 层 ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,      # 缩小 FFN
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # ---------- 2) 再建 Decoder 层 ----------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True
        )

        # ---------- 3) 共享权重 ----------
        # ① Self-Attention 共用
        dec_layer.self_attn = enc_layer.self_attn
        # ② FFN 共用
        dec_layer.linear1   = enc_layer.linear1
        dec_layer.linear2   = enc_layer.linear2
        # ③ LayerNorm 也可共享（可选）
        dec_layer.norm1     = enc_layer.norm1
        dec_layer.norm3     = enc_layer.norm2  # decoder 有 3 个 norm

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt,
                src_padding_mask=None, tgt_padding_mask=None):
        src_e = self.embed(src) * math.sqrt(self.d_model) \
                + self.pos[:, :src.size(1)]
        tgt_e = self.embed(tgt) * math.sqrt(self.d_model) \
                + self.pos[:, :tgt.size(1)]

        mem = self.encoder(src_e, src_key_padding_mask=src_padding_mask)
        out = self.decoder(
            tgt_e, mem,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device),
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.head(out)



# ========== Data Generation ==========
def generate_batch_from_digits(
        digit_string: str,
        batch_size: int = BATCH_SIZE,
        seq_len_min: int = 4,
        seq_len_max: int = 10
):
    # randomly generate sequences from the digit string
    batch_src, batch_tgt = [], []

    for _ in range(batch_size):
        cur_len = torch.randint(seq_len_min, seq_len_max + 1, (1,)).item()

        start_idx = torch.randint(0, len(digit_string) - cur_len, (1,)).item()
        digits = [int(d) for d in digit_string[start_idx:start_idx + cur_len]]

        tokens = [BOS_TOKEN] + [d + DIGIT_OFFSET for d in digits] + [EOS_TOKEN]
        padded = tokens + [PAD_TOKEN] * (MAX_SEQ_LEN - len(tokens))
        padded = padded[:MAX_SEQ_LEN]

        batch_src.append(torch.tensor(padded))
        batch_tgt.append(torch.tensor(padded))

    return torch.stack(batch_src).to(device), torch.stack(batch_tgt).to(device)


# ========== Training ==========
# model = SimpleTransformer().to(device)
model = TinySharedTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

print(f"\nTraining for {N_EPOCHS} epochs...")
print(f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
print(f"Total training samples: {BATCH_SIZE * BATCHES_PER_EPOCH * N_EPOCHS:,}")

model.train()
for epoch in range(N_EPOCHS):
    epoch_loss = 0

    for batch_idx in range(BATCHES_PER_EPOCH):
        src_data, tgt_data = generate_batch_from_digits(PI)

        # Teacher forcing: decoder input is[:-1], target is [1:]
        tgt_input = tgt_data[:, :-1]
        target = tgt_data[:, 1:]

        # Generate padding masks
        src_padding_mask = (src_data == PAD_TOKEN)
        tgt_padding_mask = (tgt_input == PAD_TOKEN)

        output = model(src_data, tgt_input,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask)

        # Reshape output for loss calculation
        loss = loss_function(output.reshape(-1, VOCAB_SIZE), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 25 == 0:
        avg_loss = epoch_loss / BATCHES_PER_EPOCH
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Avg Loss: {avg_loss:.4f}")

# ========== Evaluation Function ==========
@torch.no_grad()
def evaluate_sequence(model, digit_string, start_pos, length):
    """Evaluate the model on a sequence"""
    model.eval()

    input_digits = [int(d) for d in digit_string[start_pos:start_pos + length]]

    src_tokens = [BOS_TOKEN] + [d + DIGIT_OFFSET for d in input_digits] + [EOS_TOKEN]
    src = torch.tensor(src_tokens + [PAD_TOKEN] * (MAX_SEQ_LEN - len(src_tokens))).unsqueeze(0).to(device)
    src = src[:, :MAX_SEQ_LEN]

    src_padding_mask = (src == PAD_TOKEN)

    tgt = torch.tensor([[BOS_TOKEN]]).to(device)
    decoded_digits = []

    for _ in range(length + 5):
        output = model(src, tgt, src_padding_mask=src_padding_mask)
        next_token = output[0, -1].argmax().item()

        if next_token == EOS_TOKEN:
            break

        if DIGIT_OFFSET <= next_token < DIGIT_OFFSET + 10:
            decoded_digits.append(next_token - DIGIT_OFFSET)

        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)

        if tgt.size(1) >= MAX_SEQ_LEN:
            break

    model.train()
    return input_digits, decoded_digits

# ========== Testing ==========
print("\n" + "="*50)
print("Testing on e (unseen data):")
print("="*50)

test_positions = [0, 100, 200, 300, 400]
test_length = 7

for i, pos in enumerate(test_positions):
    input_seq, output_seq = evaluate_sequence(model, E, pos, test_length)
    match = "✓" if input_seq == output_seq else "✗"
    print(f"\nTest {i+1} (e[{pos}:{pos+test_length}]):")
    print(f"  Input:  {''.join(map(str, input_seq))}")
    print(f"  Output: {''.join(map(str, output_seq))}")
    print(f"  Match:  {match}")

print("\n" + "="*50)
print("Sanity check on π (training data, different positions):")
print("="*50)

pi_test_pos = 800
input_seq, output_seq = evaluate_sequence(model, PI, pi_test_pos, test_length)
match = "✓" if input_seq == output_seq else "✗"
print(f"\nπ[{pi_test_pos}:{pi_test_pos+test_length}]:")
print(f"  Input:  {''.join(map(str, input_seq))}")
print(f"  Output: {''.join(map(str, output_seq))}")
print(f"  Match:  {match}")