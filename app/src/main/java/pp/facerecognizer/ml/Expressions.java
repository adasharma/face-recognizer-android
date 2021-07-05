package pp.facerecognizer.ml;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

public class Expressions {
    private String LOG_TAG = "Expressions";
    private static final String MODEL_FILE = "facial_expression_model_weights.tflite";

    public static final int EMBEDDING_SIZE = 7;

    private static final int INPUT_SIZE_HEIGHT = 48;
    private static final int INPUT_SIZE_WIDTH = 48;

    private static final int BYTE_SIZE_OF_FLOAT = 4;

    private FloatBuffer inputBuffer;
    private FloatBuffer probabilityBuffer;

    private Interpreter interpreter;

    /**
     * Memory-map the model file in Assets.
     */
    private static ByteBuffer loadModelFile(AssetManager assets)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     */
    public static Expressions create(final AssetManager assetManager) {
        final Expressions f = new Expressions();

        try {
            f.interpreter = new Interpreter(loadModelFile(assetManager));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Pre-allocate buffers.
        f.inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE_HEIGHT * INPUT_SIZE_WIDTH * BYTE_SIZE_OF_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
//        f.probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1,EMBEDDING_SIZE}, DataType.FLOAT32);
        f.probabilityBuffer = ByteBuffer.allocateDirect(EMBEDDING_SIZE * BYTE_SIZE_OF_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        return f;
    }

    private Expressions() {
    }

    public FloatBuffer predict(FloatBuffer buffer) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("getEmotions");

        // Run the inference call.
        Trace.beginSection("run");
        probabilityBuffer.rewind();
        float[][] emotion=new float[1][7];
        interpreter.run(buffer, emotion);
        float emotion_v=(float)Array.get(Array.get(emotion,0),0);
        Log.d("facial_expression","Output:  "+ emotion_v);
        String emotion_s=get_emotion_text(emotion_v);
        Log.d("facial_expression","Output:  "+ emotion_s);
        probabilityBuffer.flip();
        Trace.endSection();

        Trace.endSection();
        return probabilityBuffer;
    }

    public void close() {
        interpreter.close();
    }

    private String get_emotion_text(float emotion_v) {
        // create an empty string
        String val="";
        // use if statement to determine val
        // You can change starting value and ending value to get better result
        // Like

        if(emotion_v>=0 & emotion_v<0.5){
            val="Surprise";
        }
        else if(emotion_v>=0.5 & emotion_v <1.5){
            val="Fear";
        }
        else if(emotion_v>=1.5 & emotion_v <2.5){
            val="Angry";
        }
        else if(emotion_v>=2.5 & emotion_v <3.5){
            val="Neutral";
        }
        else if(emotion_v>=3.5 & emotion_v <4.5){
            val="Sad";
        }
        else if(emotion_v>=4.5 & emotion_v <5.5){
            val="Disgust";
        }
        else {
            val="Happy";
        }
        return val;
    }

}
