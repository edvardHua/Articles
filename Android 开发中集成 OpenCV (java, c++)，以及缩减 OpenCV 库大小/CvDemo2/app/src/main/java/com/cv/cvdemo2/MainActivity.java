package com.cv.cvdemo2;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

	private ImageView iv;

	// Used to load the 'native-lib' library on application startup.
	static {
		System.loadLibrary("opencv_java3");
		System.loadLibrary("native-lib");
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		iv = findViewById(R.id.display_img);

		Bitmap image = BitmapFactory.decodeResource(getResources(), R.drawable.lenna).copy(Bitmap.Config.ARGB_8888, true);
		int width = image.getWidth();
		int height = image.getHeight();
		int[] pixel = new int[width * height];
		image.getPixels(pixel, 0, width, 0, 0, width, height);
		int[] grayPixels = convertToGray(pixel, width, height);

		Bitmap grayBp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
		grayBp.setPixels(grayPixels, 0, width, 0, 0, width, height);
		iv.setImageBitmap(grayBp);
	}

	public native int[] convertToGray(int[] imgData, int width, int height);
}
