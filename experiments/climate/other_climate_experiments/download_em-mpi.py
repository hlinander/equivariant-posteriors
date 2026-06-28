from huggingface_hub import snapshot_download # Requires pip install huggingface_hub

snapshot_download(repo_id='blutjens/em-mpi',
  repo_type='dataset', 
  local_dir='/proj/heal_pangu/users/x_tagty/em-mpi', # Local path to data 
)