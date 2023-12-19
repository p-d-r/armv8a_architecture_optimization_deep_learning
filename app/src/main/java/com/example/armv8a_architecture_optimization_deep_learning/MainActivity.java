package com.example.armv8a_architecture_optimization_deep_learning;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;
import android.content.res.AssetManager;
import com.chaquo.python.PyException;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import com.example.armv8a_architecture_optimization_deep_learning.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    // Load the native library
    static {
        System.loadLibrary("armv8a_architecture_optimization_deep_learning");
    }

    // Declare the native method
    //public native void load_model(AssetManager assetManager);
    public native void profiler_call();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        // Call the native methods
        //load_model(getResources().getAssets());
        profiler_call();
    }
}