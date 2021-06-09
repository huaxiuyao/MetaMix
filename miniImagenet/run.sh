# meta-training
# 1-shot
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=1 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --test_set=1 --mix --shuffle
# 5-shot
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --test_set=1 --mix --shuffle

# meta-testing
# 1-shot
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=1 --update_batch_size_eval=1 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --test_set=1 --mix --shuffle --train=0 --test_epoch=xxx
# 5-shot
python main.py --datasource=miniimagenet --metatrain_iterations=50000 --meta_lr=0.001 --meta_batch_size=4 --update_batch_size=5 --update_batch_size_eval=5 --num_classes=5 --datadir=xxx --logdir=xxx --num_filters=32 --test_set=1 --mix --shuffle --train=0 --test_epoch=xxx