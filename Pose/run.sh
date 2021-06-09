# meta-training
# 10-shot
python main.py -datasource=pose --data_dir=xxx --logdir=xxx --data=train_data_ins.pkl,val_data_ins.pkl --update_lr=0.01 --meta_lr=0.001 --update_batch_size=10 --meta_batch_size=10 --num_updates=5 --test_num_updates=20 --mix=True --trial=1
# 15-shot
python main.py -datasource=pose --data_dir=xxx --logdir=xxx --data=train_data_ins.pkl,val_data_ins.pkl --update_lr=0.01 --meta_lr=0.001 --update_batch_size=15 --meta_batch_size=10 --num_updates=5 --test_num_updates=20 --mix=True --trial=1

# meta-testing
# 10-shot
python main.py -datasource=pose --data_dir=xxx --logdir=xxx --data=train_data_ins.pkl,val_data_ins.pkl --update_lr=0.01 --meta_lr=0.001 --update_batch_size=10 --meta_batch_size=10 --num_updates=5 --test_num_updates=20 --mix=True --trial=1 --train=False --test_epoch=xxx
# 15-shot
python main.py -datasource=pose --data_dir=xxx --logdir=xxx --data=train_data_ins.pkl,val_data_ins.pkl --update_lr=0.01 --meta_lr=0.001 --update_batch_size=15 --meta_batch_size=10 --num_updates=5 --test_num_updates=20 --mix=True --trial=1 --train=False --test_epoch=xxx