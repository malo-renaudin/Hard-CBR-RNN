from nounpp_viz_eval import load_tokenizer, NounPPDataset, collate_fn_nounpp, CBR_RNN, analyze_and_plot, SimpleTransformer, run_all_diagnostics
from viz_attn import load_model
from torch.utils.data import Dataset, DataLoader
import torch
from grid_search import WordTokenizer

no_gs = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_001/lightning_logs/version_1364336/checkpoints/epoch=49-step=309500.ckpt'
check_0_1_exp = '/scratch2/mrenaudin/Hard-CBR-RNN/job_005/lightning_logs/version_1198536/checkpoints/epoch=49-step=565950.ckpt'
check_0_5_exp = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_000/lightning_logs/version_1934821/checkpoints/epoch=49-step=309500.ckpt'
check_0_1_cosine = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_003/lightning_logs/version_1934824/checkpoints/epoch=49-step=309500.ckpt'
check_0_5_cosine = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_004/lightning_logs/version_1934763/checkpoints/epoch=49-step=309500.ckpt'
check_0_1_linear = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_001/lightning_logs/version_1934822/checkpoints/epoch=49-step=309500.ckpt'
check_0_5_linear = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_002/lightning_logs/version_1934823/checkpoints/epoch=49-step=309500.ckpt'
big_dim = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_001/lightning_logs/version_1850825/checkpoints/epoch=49-step=309500.ckpt'
check_transformer_exp_01 = '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_005/lightning_logs/version_1356206/checkpoints/epoch=49-step=309500.ckpt'
check_big_transformer_exp_0_1 = '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_005/lightning_logs/version_1851789/checkpoints/epoch=49-step=309500.ckpt'
check_big_cbr_exp_0_1 = '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_005/lightning_logs/version_1850865/checkpoints/epoch=49-step=309500.ckpt'
check_big_transformer_no_gs = '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_001/lightning_logs/version_1851738/checkpoints/epoch=49-step=309500.ckpt'
transformer_1024_8_heads = '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_003/lightning_logs/version_1851787/checkpoints/epoch=49-step=309500.ckpt'
lstm = '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_003/lightning_logs/version_1851441/checkpoints/epoch=49-step=309500.ckpt'

data_dir = "cbr_lightning/wikitext-103-raw"


checkpoint = torch.load(transformer_1024_8_heads, map_location='cuda')
state_dict = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('model.', '')  # remove 'model.' prefix
    new_state_dict[name] = v
# model = CBR_RNN(49999, 512, 512, 1, 1, 0)
model = SimpleTransformer(49999, d_model=1024, nhead=8, num_layers=2)
# from simple_lstm import SimpleLSTM_LM
# def load_trained_lstm(checkpoint_path):
#     model=SimpleLSTM_LM.load_from_checkpoint(checkpoint_path)
#     model.eval()
#     return model
# model = load_trained_lstm(lstm)
model.load_state_dict(new_state_dict)
# stoi, itos = load_tokenizer('tokenizer.json') 
tokenizer = WordTokenizer.load("tokenizer.json")
test_dataset = NounPPDataset('nounpp.txt', tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=1000, collate_fn=collate_fn_nounpp)

results = analyze_and_plot(model, test_dataloader, temperature=1, 
                     use_gumbel=False, save_dir=None, seed=42)
run_all_diagnostics(model, test_dataloader, tokenizer, device='cuda')

# print(f"'remember' in tokenizer: {'remember' in stoi}")
