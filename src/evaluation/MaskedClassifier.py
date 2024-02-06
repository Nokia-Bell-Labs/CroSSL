import random
import itertools
import tensorflow as tf
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
def get_classifier_model(dropout, class_size, input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    #x = tf.keras.layers.Dense(32, activation="relu")(x)
    #x = tf.keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    classifier_model = tf.keras.layers.Dense(class_size, activation="softmax", name="classifier_last_dense")(x)

    return tf.keras.Model(inputs, classifier_model)

class MaskedClassifier(tf.keras.Model):
    def __init__(self, embedding_model, projection_model, classifier, ds_name, no_device=3, missing=True,**kwargs):
        super().__init__()
        self.encoder = embedding_model
        self.projector = projection_model
        self.classifier = classifier
        self.subset_idx = [()]
        self.dataset = ds_name
        if missing:
            self.subset_idx = self.generate_subsets(no_device)


    def generate_subsets(self,n):
        subsets = []
        s = list(range(0, n))
        for i in range(0, n):
            subsets.extend(list(itertools.combinations(s, i)))
        self.device_subset = subsets
        subsets_idx =[]
        for s in subsets:
            sensor_list=[]
            for d in s:
                if self.dataset in ['sleepedf','wesad']:
                    sensor_list.append(d)
                elif self.dataset == 'pamap2':
                    sensor_list.append(d*2)
                    sensor_list.append(1+d * 2)
                elif self.dataset == 'pamap2_hr':
                    sensor_list.append(d*2)
                    if d<3:
                        sensor_list.append(1+d * 2)
            subsets_idx.extend([sensor_list])
        return subsets_idx

    def get_masked_tensor(self,tensor, missing_device):

        #print("missing device: ", missing_device)
        tensor_shape = tensor.shape
        for md in missing_device:

            #ind = md*2 #index of sensor #only PAMAP *2
            # Define the indices to be masked
            indices = tf.constant([[i, md] for i in range(tensor_shape[0])] )
            # Mask the values in the tensor at the specified indices
            tensor = tf.tensor_scatter_nd_update(tensor, indices, tf.zeros((tensor_shape[0], tensor_shape[-1])))

            #indices = tf.constant([[i, ind+1] for i in range(tensor_shape[0])]) #only PAMAP
            # Mask the values in the tensor at the specified indices
            #tensor = tf.tensor_scatter_nd_update(tensor, indices, tf.zeros((tensor_shape[0], tensor_shape[-1]))) #only PAMAP
        #print(tensor)
        return tensor

    def call(self, inputs, training=None):
        #print(" --------------->>>>>> inside CALL <<<<<<<<----------------")
        x = self.encoder(inputs, training=training)
        x = tf.stack(x)
        x = tf.transpose(x, (1, 0, 2))
        x = self.get_masked_tensor(x, self.subset_idx[random.randint(0, len(self.subset_idx)-1)])
        x = self.projector(x, training=training)
        out = self.classifier(x, training=training)
        return out

    def train_step(self, data):
        with tf.GradientTape() as tape:
            inputs, labels = data
            modality_embeddings = self.encoder(inputs, training=False)
            modality_embeddings = tf.stack(modality_embeddings) #(8, 2, 64)
            # dim_size = len(modality_embeddings)
            modality_embeddings = tf.transpose(modality_embeddings, (1, 0, 2))  # Batch x Mod x code_size
            modality_embeddings = self.get_masked_tensor(modality_embeddings,self.subset_idx[random.randint(0, len(self.subset_idx)-1)])

            rep = self.projector(modality_embeddings) # aggregator
            logits = self.classifier(rep)

            loss = self.compiled_loss(labels, logits)
            loss += sum(self.losses)
            # loss += sum(self.projector.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def custom_evaluate(self, data):
        #data = data.map(unpack_fn)
        #x = [d for d,l in data]
        #y = [l for d,l in data]
        x,y = data
        f1 = []
        accuracy = []
        auc = []
        for missing_idx in self.device_subset: #  [[0], [1], [2], [0,1], [1,2], [0,2], []]
            x = self.encoder(x, training=False)
            x = tf.stack(x)
            x = tf.transpose(x, (1, 0, 2))
            x = self.get_masked_tensor(x, missing_idx)
            x = self.projector(x, training=False)
            logits = self.classifier(x, training=False)
            # Calculate Accuracy
            acc = tf.keras.metrics.categorical_accuracy(y, logits)
            accuracy.append(acc)
            # Calculate F1-score
            f1 = f1_score(tf.argmax(y, axis=1), tf.argmax(logits, axis=1), average='macro')
            f1.append(f1)
            # Calculate AUC

            auc.append(roc_auc_score(y, logits))
            print("missing sensors {} -- f1 : {} / acc : {}".format( missing_idx,f1,acc) )
        return {'f1_score': tf.reduce_mean(f1), 'auc': tf.reduce_mean(auc), 'categorical_accuracy': tf.reduce_mean(accuracy)}



# Comparisons:
# fine-tuning and inference with missing modalities
# fine-tuning with full data and inference with missing modalities
# fine-tuning and inference with
