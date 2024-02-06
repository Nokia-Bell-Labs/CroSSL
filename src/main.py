from src.model_utils.common_utilities import *
from exp_setup import *
from evaluation.MaskedClassifier import get_classifier_model, MaskedClassifier
from evaluation.classifier import get_classifier
from model_utils.model import get_encoder, get_projector
from trainer import *

MASKED_CLS =False
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)

if __name__ == '__main__':
    args = get_parser().parse_args()
    working_dir, encoder_name, log_dir, ds_info = setup_system(args)
    report_file_path = os.path.join(working_dir, "results", "report.csv")

    encoder = get_encoder(ds_info['win_size'], ds_info['mod_name'], ds_info['mod_dim'],
                          args.code_size, modality_filters=args.filters, l2_rate=1e-4)
    aggregator = get_projector(args.code_size, args.proj_size, len(ds_info['mod_dim']))
    ssl_loss_value = 0
    if args.mode in ['ssl', 'fine']:
        cbs, optimizer_ssl, loss_fn, ssl_data = setup_ssl_training(log_dir, args)
        ssl_model = cm_model(ds_info, code_size=args.code_size, proj_size=args.proj_size,
                             modality_filters=args.filters, l2_rate=1e-4, coverage=args.coverage, masking=args.masking)
        ssl_model.compile(optimizer=optimizer_ssl, loss=loss_fn,
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        ssl_model.fit(ssl_data, batch_size=args.batch_size, epochs=args.epoch, verbose=args.verbosity, callbacks=cbs)
        ssl_loss_value = ssl_model.history.history['loss'][0]
        # Evaluation
        encoder = ssl_model.encoder
        aggregator = ssl_model.projector

    if args.mode == "ssl":
        encoder.trainable = False
        for l in encoder.layers:
            l.trainable = False

        aggregator.trainable = False
        for l in aggregator.layers:
            l.trainable = False

    #tsne_visualize(args.datapath, args.dataset, working_dir, base_encoder=ssl_model, exp_id=encoder_name, mode='ssl')

    cbs, optimizer_cls, metrics, trn_data, val_data, tst_data = setup_cls_training(log_dir, args)
    if MASKED_CLS:
        classifier = get_classifier_model(dropout=0.1, class_size=ds_info['class_size'], input_shape=(32))
        linear_cl = MaskedClassifier(encoder, aggregator, classifier, ds_name=ds_info['ds_name'], no_device=ds_info['num_device'],
                                     missing=args.missing_ft)
    else:
        linear_cl = get_classifier(encoder, encoder_only=True, class_size=ds_info['class_size'], projector=aggregator)
    linear_cl.compile(loss="categorical_crossentropy", metrics=metrics,
                         optimizer=optimizer_cls)

    history = linear_cl.fit(trn_data,
                               validation_data=val_data,
                               batch_size=args.batch_size,
                               epochs=args.epoch,
                               callbacks=cbs)


    #tsne_visualize(args.datapath, args.dataset, working_dir, base_encoder=base_model, exp_id=encoder_name, mode="fine_"+args.mode)

    train_score = linear_cl.evaluate(trn_data, verbose=args.verbosity)
    print(" >>>>>> TRAIN SCORE : <<<<<<<<<\n", train_score)
    test_score = linear_cl.evaluate(tst_data, verbose=args.verbosity)
    print(" >>>>>> TEST SCORE : <<<<<<<<<\n", test_score)

