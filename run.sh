datapath=/content/drive/MyDrive/Dataset
datasets=('CMI')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python /content/ThanhNet/main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mci \
--log_project MCIAD_Results \
--results_path results \
--run_name run \
net \
-le layer1 \
--pretrain_embed_dimension 1024 \
--target_embed_dimension 1024 \
--patchsize 3 \
--meta_epochs 60 \
--embedding_size 64 \
--gan_epochs 5 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 "${dataset_flags[@]}" cmi $datapath
