package com.cv.demo1;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity {

	private ImageView iv;

	static {
		// 加载对应的 so 文件，需要去头 lib，去尾 .so
		System.loadLibrary("opencv_java3");
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		iv = findViewById(R.id.display_img);

		Bitmap bitmap = BitmapFactory.decodeResource(MainActivity.this.getResources(), R.drawable.lenna).copy(Bitmap.Config.ARGB_8888, true);
		Mat mat = new Mat();
		Utils.bitmapToMat(bitmap, mat);

		Mat grayMat = new Mat();
		Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);

		Bitmap grayBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(grayMat, grayBitmap);
		iv.setImageBitmap(grayBitmap);
	}
}
