import React, { useState, useEffect } from 'react';
import { View, Text, Image, TextInput, Button, StyleSheet, Alert } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { fetch as tfFetch, decodeJpeg } from '@tensorflow/tfjs-react-native';

const App: React.FC = () => {
  const [isTfReady, setIsTfReady] = useState(false);
  const [model, setModel] = useState<any>(null); // Placeholder for the model
  const [result, setResult] = useState<string | null>(null);
  const [imageUri, setImageUri] = useState(
    'https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvcHUyMzMxNjM2LWltYWdlLWt3dnk3dzV3LmpwZw.jpg'
  );

  useEffect(() => {
    const initTensorFlow = async () => {
      await tf.ready();
      setIsTfReady(true);

      // Load the MobileNet model
      const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
      setModel(model);
    };
    initTensorFlow();
  }, []);

  const loadAndClassifyImage = async () => {
    try {
      if (!isTfReady || !model) {
        Alert.alert('TensorFlow.js or model is not ready');
        return;
      }

      // Fetch image
      const response = await tfFetch(imageUri, {}, { isBinary: true });
      const imageDataArrayBuffer = await response.arrayBuffer();
      const imageData = new Uint8Array(imageDataArrayBuffer);

      // Decode image to tensor
      const imageTensor = decodeJpeg(imageData);

      // Preprocess image tensor
      const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]).toFloat();
      const normalizedImage = resizedImage.div(tf.scalar(255)).expandDims(0);

      // Predict
      const predictions = await model.predict(normalizedImage) as tf.Tensor;
      const topK = predictions.topk(5); // Top 5 predictions

      // Extract values and indices from topK
      const topKValues = await topK.values.array() as number[][];
      const topKIndices = await topK.indices.array() as number[][];

      // Format results
      const resultString = topKValues[0].map((value, index) => 
        `Class ${topKIndices[0][index]}: ${(value * 100).toFixed(2)}%`
      ).join('\n');

      setResult(resultString);
    } catch (err) {
      console.error('Error loading or processing image:', err);
      Alert.alert('Error', 'Failed to load or process image.');
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Enter image URL"
        value={imageUri}
        onChangeText={setImageUri}
      />
      <Button title="Load and Classify Image" onPress={loadAndClassifyImage} />
      {imageUri ? (
        <Image
          source={{ uri: imageUri }}
          style={styles.image}
        />
      ) : null}
      {!isTfReady && <Text>Loading TensorFlow.js...</Text>}
      {result && <Text>{result}</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'white',
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
    width: '100%',
  },
  image: {
    width: 200,
    height: 200,
    marginTop: 20,
  },
});

export default App;
