datapath=/content/drive/MyDrive/Dataset
datasets=('CMI')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python /content/TNet/main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mci \
--log_project MCIAD_Results \
--results_path results \
--run_name run \
net \
-le layer1 \
-le layer2 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 60 \
--embedding_size 60 \
--gan_epochs 5 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 "${dataset_flags[@]}" cmi $datapath
