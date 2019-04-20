//
//  ViewController.swift
//  Obtain3DCoordinate
//
//  Created by edvard on 2019/4/20.
//  Copyright © 2019 edvard. All rights reserved.
//

import UIKit
import Accelerate
import AVFoundation

class ViewController: UIViewController, AVCaptureDataOutputSynchronizerDelegate {

    var displayType = "video"
    @IBOutlet weak var touchCoord: UILabel!
    @IBOutlet weak var preview: UIImageView!
    @IBOutlet weak var switchType: UIButton!
    
    // Camera related parameters
    let videoDataOutput = AVCaptureVideoDataOutput()
    let depthDataOutput = AVCaptureDepthDataOutput()
    var videoDeviceInput: AVCaptureDeviceInput!
    var outputSynchronizer: AVCaptureDataOutputSynchronizer?
    var depthPixelBuffer: CVPixelBuffer?
    let session = AVCaptureSession()
    let videoDeviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInTrueDepthCamera], mediaType: .video, position: .front)
    
    private let dataOutputQueue = DispatchQueue(label: "com.cameraDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    // intrinsic parameters
    var refWidth: Float?
    var refHeight: Float?
    var camOx: Float?
    var camOy: Float?
    var camFx: Float?
    var camFy: Float?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        configureSession()
    }
    
    @IBAction func previewTypeChange(_ sender: Any) {
        if displayType == "video" {
            switchType.setTitle("切换为彩色图", for: .normal)
            displayType = "depth"
        } else if displayType == "depth" {
            switchType.setTitle("切换为深度图", for: .normal)
            displayType = "video"
        }
        print(displayType)
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        print(touches.count)
        let touchPoint = (touches as NSSet).allObjects[0] as! UITouch
        let coord = touchPoint.location(in: self.preview)
        let viewContent = self.preview.bounds
        let xRatio = Float(coord.x / viewContent.size.width)
        let yRatio = Float(coord.y / viewContent.size.height)
        let realZ = getDepth(from: depthPixelBuffer!, atXRatio: xRatio, atYRatio: yRatio)
        let realX = (xRatio * refWidth! - camOx!) * realZ / camFx!
        let realY = (yRatio * refHeight! - camOy!) * realZ / camFy!
        DispatchQueue.main.async {
            self.touchCoord.text = String.localizedStringWithFormat("X = %.2f cm, Y = %.2f cm, Z = %.2f cm", realX, realY, realZ)
        }
    }
    
    private func getDepth(from depthPixelBuffer: CVPixelBuffer, atXRatio: Float, atYRatio: Float) -> Float {
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        let depthWidth = CVPixelBufferGetWidth(depthPixelBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthPixelBuffer)
        let rowData = CVPixelBufferGetBaseAddress(depthPixelBuffer)! + Int(atYRatio * Float(depthHeight)) *  CVPixelBufferGetBytesPerRow(depthPixelBuffer)
        var f16Pixel = rowData.assumingMemoryBound(to: UInt16.self)[Int(atXRatio * Float(depthWidth))]
        var f32Pixel = Float(0.0)
        var src = vImage_Buffer(data: &f16Pixel, height: 1, width: 1, rowBytes: 2)
        var dst = vImage_Buffer(data: &f32Pixel, height: 1, width: 1, rowBytes: 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        let depth = f32Pixel * 100
        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, CVPixelBufferLockFlags(rawValue: 1))
        return depth
    }
    
    private func configureSession() {
        
        let defaultVideoDevice: AVCaptureDevice? = videoDeviceDiscoverySession.devices.first
        guard let videoDevice = defaultVideoDevice else {
            print("Could not find any video device")
            return
        }
        
        do {
            videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = AVCaptureSession.Preset.photo
        
        guard session.canAddInput(videoDeviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        session.addInput(videoDeviceInput)
        
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        depthDataOutput.alwaysDiscardsLateDepthData = true
        
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            let videoConnection = videoDataOutput.connection(with: .video)
            videoConnection?.videoOrientation = .portrait
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        
        if session.canAddOutput(depthDataOutput) {
            session.addOutput(depthDataOutput)
            depthDataOutput.isFilteringEnabled = true
            if let connection = depthDataOutput.connection(with: .depthData) {
                connection.isEnabled = true
                connection.videoOrientation = .portrait
            } else {
                print("No AVCaptureConnection")
            }
        } else {
            print("Could not add depth data output to the session")
            session.commitConfiguration()
            return
        }
        
        // Search for highest resolution with half-point depth values
        let depthFormats = videoDevice.activeFormat.supportedDepthDataFormats
        let filtered = depthFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
        })
        let selectedFormat = filtered.max(by: {
            first, second in CMVideoFormatDescriptionGetDimensions(first.formatDescription).width < CMVideoFormatDescriptionGetDimensions(second.formatDescription).width
        })
        
        do {
            try videoDevice.lockForConfiguration()
            videoDevice.activeDepthDataFormat = selectedFormat
            videoDevice.unlockForConfiguration()
        } catch {
            print("Could not lock device for configuration: \(error)")
            session.commitConfiguration()
            return
        }
        
        outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        outputSynchronizer!.setDelegate(self, queue: dataOutputQueue)
        session.commitConfiguration()
        
        self.session.startRunning()
    }
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        guard let syncedDepthData: AVCaptureSynchronizedDepthData =
            synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData, let syncedVideoData: AVCaptureSynchronizedSampleBufferData =
            synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData else { return }
        
        if syncedDepthData.depthDataWasDropped || syncedVideoData.sampleBufferWasDropped {
            return
        }
        // fixed value
        let intrinsicMartix = syncedDepthData.depthData.cameraCalibrationData?.intrinsicMatrix
        let refenceDimension = syncedDepthData.depthData.cameraCalibrationData?.intrinsicMatrixReferenceDimensions
        self.camFx = intrinsicMartix![0][0]
        self.camFy = intrinsicMartix![1][1]
        self.camOx = intrinsicMartix![0][2]
        self.camOy = intrinsicMartix![1][2]
        self.refWidth = Float(refenceDimension!.width)
        self.refHeight = Float(refenceDimension!.height)
        
        var displayPixelBuffer:CVPixelBuffer?
        self.depthPixelBuffer = syncedDepthData.depthData.depthDataMap
        if displayType == "video" {
            displayPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer)
        }else if displayType == "depth"{
            displayPixelBuffer = syncedDepthData.depthData.depthDataMap
        }

        let image = CIImage(cvPixelBuffer: displayPixelBuffer!)
        let displayImage = UIImage(ciImage: image)
        
        DispatchQueue.main.async {
            self.preview.image = displayImage
        }
    }
    
    
}

