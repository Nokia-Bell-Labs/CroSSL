HHAROpt = {
    'name': 'hhar',
    'batch_size': 16,
    'batch_size_target': 8,
    'seq_len': 256,
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0.001,

    'dataset_path': './dataset/hhar',
    'dataset_name': './dataset/hhar_minmax_scaling_all.csv',

    ###---- 24 available domains for evaluation
    'users': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],  # 9 users
    'models': ['nexus4', 's3', 's3mini', 'lgwatch'],
    'devices': ['lgwatch_1', 'lgwatch_2', 'gear_1', 'gear_2', 'nexus4_1', 'nexus4_2',
                's3_1', 's3_2', 's3mini_1', 's3mini_2'],

    'classes': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'],
    'num_class': 6,

'lstm_units': 32,
    'cnn_filters': 3,
    'num_lstm_layers': 1,
    'patience': 20,
    'F': 32,
    'D': 10
}

RealWorldOpt = {
    'seq_len': 150,
    'input_dim': 6,
    'devices': ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin'],
    # 'devices': ['forearm', 'thigh'],
    'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13','S14', 'S15'],
    # 'users': ['S1', 'S2'],
    'classes': ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking'],
    'num_class': 8,
    'dataset_path': './',
    'dataset_name': 'realworld-3.0-0.0.dat',
    'batch_size': 16,

    'lstm_units': 32,
    'cnn_filters': 3,
    'num_lstm_layers': 1,
    'patience': 20,
    'F': 32,
    'D': 10
}

OpportunityOpt = {
    # 'sensors': ['accelerometer', 'gyroscope', 'magnetic', 'quaternion', 'ic3_eu', 'ic3_nav', 'ic3_body',
    #             'ic3_angvelbody', 'ic3_angvelnav', 'ic3_compass', 'reed_switch', 'location_tag'],
    # 'devices': ['body_rkn^', 'body_hip', 'body_lua^', 'body_rua_', 'body_lh', 'body_back', 'body_rkn_', 'body_rwr',
    #             'body_rua^', 'body_lua_', 'body_lwr', 'body_rh',
    #             'imu_back', 'imu_rua', 'imu_rla', 'imu_lua', 'imu_lla', 'imu_lshoe', 'imu_rshoe',
    #             'obj_acc_cup', 'obj_acc_salami', 'obj_acc_water', 'obj_acc_cheese',
    #             'obj_acc_bread', 'obj_acc_knife1', 'obj_acc_milk', 'obj_acc_spoon', 'obj_acc_sugar', 'obj_acc_knife2',
    #             'obj_acc_plate', 'obj_acc_glass', 'obj_dishwasher', 'obj_fridge', 'obj_middledrawer', 'obj_upperdrawer',
    #             'obj_lowerdrawer', 'obj_acc_door1', 'obj_acc_lazychair', 'obj_acc_door2',
    #             'loc_tag1', 'loc_tag2', 'loc_tag3', 'loc_tag4'],
    'devices': ['back', 'lla', 'rshoe', 'rua', 'lshoe'],
    'users': ['S1', 'S2', 'S3', 'S4'],
    'sessions': ['ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5', 'Drill'],
    'classes': [0, 1, 2, 3],
    'num_class': 4,
    'batch_size': 16,
    'dataset_path': './',
    'dataset_name': 'opportunity-2.0-0.0.dat',
    'seq_len': 60,
    'input_dim': 6,
    'lstm_units': 32,
    'cnn_filters': 3,
    'num_lstm_layers': 1,
    'patience': 20,
    'F': 32,
    'D': 10
}

PAMAP2Opt = {
    'dataset_name': 'pamap2adl-2.0-0.0.dat',
    'devices': ['hand', 'chest', 'ankle'],
    'users': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'],
    'classes': ['lying', 'sitting', 'standing', 'walking', 'running',
                'cycling', 'nordic_walking', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning',
                'ironing', 'rope_jumping'],
    'num_class': 12,
    'dataset_path': './',
    'seq_len': 200,
    'input_dim': 6,

    'lstm_units': 32,
    'cnn_filters': 3,
    'num_lstm_layers': 1,
    'patience': 20,
    'F': 32,
    'D': 10
}

