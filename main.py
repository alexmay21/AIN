import argparse
from time import time
import tensorflow as tf
from LoadData import Dataset
import os
from AIN import AIN


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--path', nargs='?', default='./dataset/processed/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Food',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors of user and item.')
    parser.add_argument('--context_hidden_factor', type=int, default=64,
                        help='Number of hidden factors of context.')
    parser.add_argument('--effect_hidden_factor', type=int, default=64,
                        help='Number of hidden factors of effect.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--mlp1_layers', type=int, default=2,
                        help='Number of layer in Interaction-Centric Module.')
    parser.add_argument('--mlp2_layers', type=int, default=1,
                        help='Number of layer in User/Item-Centric Module.')
    parser.add_argument('--mlp1_hidden_size', type=int, default=128,
                        help="Dimension of the hidden size of Interaction-Centric Module.")
    parser.add_argument('--mlp2_hidden_size', type=int, default=128,
                        help="Dimension of the hidden size of User/Item-Centric Module.")
    parser.add_argument('--keep_prob', type=float, default=0.7,
                        help='Keep probability.')

    return parser.parse_args()


if __name__ == '__main__':
    # # GPU Config
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    args = parse_args()

    # load parameters
    epoch = args.epoch
    batch_size = args.batch_size
    hidden_factor = args.hidden_factor
    context_hidden_factor = args.context_hidden_factor
    effect_hidden_factor = args.effect_hidden_factor
    learning_rate = args.learning_rate
    mlp1_layers = args.mlp1_layers
    mlp2_layers = args.mlp2_layers
    mlp1_hidden_size = args.mlp1_hidden_size
    mlp2_hidden_size = args.mlp2_hidden_size
    keep_prob = args.keep_prob

    # load dataset
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    training_set, validation_set, testing_set = dataset.training_set, dataset.validation_set, dataset.testing_set

    # statistics of the dataset
    num_users = dataset.num_users
    num_items = dataset.num_items
    num_ratings = dataset.num_ratings
    dims = dataset.dims
    num_context_dims = len(dims) - 2
    num_context = dataset.num_context
    global_mean = dataset.global_mean

    print("=" * 30)
    # print the statistic information of current used dataset
    print("Load dataset done [%.1f s]. #user=%d, #item=%d, #ratings=%d"
          % (time() - t1, num_users, num_items, num_ratings))

    # print the information of training/validation/testing set
    print("Split dataset done. #traning=%d, #validation=%d, #testing=%d"
          % (len(training_set[1]), len(validation_set[1]), len(testing_set[1])))

    print("\nDataset statistic:")
    print("num_users:", num_users)
    print("num_items:", num_items)
    print("num_ratings:", num_ratings)
    print("num_context_dims:", num_context_dims)
    print("num_context:", num_context)
    print("dims:", dims)

    print("\nModel parameters:\ndataset=%s\nepoch=%d\nbatch_size=%d\n"
          "hidden_factor=%d, context_hidden_factor=%d, effect_hidden_factor=%d\n"
          "learning_rate=%.4f\n"
          "interaction-centric module layers=%d, interaction-centric module hidden size=%d\n"
          "user/item-centric module layers=%d, user/item-centric module hidden size=%d"
          % (args.dataset, epoch, batch_size,
             hidden_factor, context_hidden_factor, effect_hidden_factor, learning_rate,
             mlp1_layers, mlp1_hidden_size, mlp2_layers, mlp2_hidden_size))
    print("=" * 30)

    start_time = time()

    with tf.Session(config=config) as sess:
        model = AIN(sess, dims, num_context, global_mean, epoch, hidden_factor, context_hidden_factor,
                    effect_hidden_factor, keep_prob, learning_rate, batch_size, mlp1_layers,
                    mlp2_layers, mlp1_hidden_size, mlp2_hidden_size, args.dataset)
        valid_rmse_record, valid_mae_record, test_rmse_record, test_mae_record = model.train(training_set, validation_set, testing_set)

        best_valid_rmse = min(valid_rmse_record)

        best_epoch_rmse = valid_rmse_record.index(best_valid_rmse)

        print("\n\n\n")
        print("Best Iter= %d: valid_rmse = %.4f, valid_mae = %.4f, test_rmse = %.4f, test_mae = %.4f"
              % (best_epoch_rmse + 1, valid_rmse_record[best_epoch_rmse], valid_mae_record[best_epoch_rmse], test_rmse_record[best_epoch_rmse], test_mae_record[best_epoch_rmse]))