import tensorflow as tf
import numpy as np
import cv2

class TensorflowLiteClassificationModel:
    def __init__(self,model_path,labels,image_size = 256):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_detals = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size = image_size
    def run_from_filepath(self,image_path):
        input_data_type = self._input_details[0]['dtype']
        img = cv2.imread(image_path)
        image = np.array(cv2.resize(img,(self.image_size,self.image_size)),dtype = np.float32)
        image = image / 255
        if image.shape == (1,224,224):
            image = np.stack(image*3,axis = 0)
        return self.run(image)
    def run(self,image):
        image = np.expand_dims(image,axis = 0)
        self.interpreter.set_tensor(self._input_details[0]['index'],image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_detals[0]['index'])
        probabilities = np.array(tflite_interpreter_output[0])

        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i],float(probability)])
        return sorted(label_to_probabilities,key = lambda element: element[1])