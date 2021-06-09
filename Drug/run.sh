# meta-training
python main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=1 --datadir=xxx --logdir=xxx --mixup
python main.py --datasource=drug --metatrain_iterations=50 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=2 --datadir=xxx --logdir=xxx --mixup
python main.py --datasource=drug --metatrain_iterations=30 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=3 --datadir=xxx --logdir=xxx --mixup
python main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=4 --datadir=xxx --logdir=xxx --mixup

# meta-training
python main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=1 --datadir=xxx --logdir=xxx --mixup --train=0 --test_epoch=xxx
python main.py --datasource=drug --metatrain_iterations=50 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=2 --datadir=xxx --logdir=xxx --mixup --train=0 --test_epoch=xxx
python main.py --datasource=drug --metatrain_iterations=30 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=3 --datadir=xxx --logdir=xxx --mixup --train=0 --test_epoch=xxx
python main.py --datasource=drug --metatrain_iterations=20 --update_lr=0.005 --meta_lr=0.001 --num_updates=5 --test_num_updates=5 --trial=1 --drug_group=4 --datadir=xxx --logdir=xxx --mixup --train=0 --test_epoch=xxx