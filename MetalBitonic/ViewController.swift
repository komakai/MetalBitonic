//
//  ViewController.swift
//  MetalBitonic
//
//  Created by Giles Payne on 2024/12/03.
//

import UIKit

let bufferLength = 1024 * 512
let bufferSize = MemoryLayout<Float>.stride * bufferLength

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    func randomFloat() -> Float
    {
        return Float.random(in: 0..<1)
    }

    func randomInt32() -> Int32
    {
        return Int32.random(in: Int32.min..<Int32.max)
    }

    func randomInt32SmallPositive() -> Int32
    {
        return Int32.random(in: 0..<256)
    }

    @IBAction func onAdd(_ sender: Any) {
        let device = MTLCreateSystemDefaultDevice()
        let defaultLibrary = device?.makeDefaultLibrary()
        let addFunction = defaultLibrary?.makeFunction(name: "add_arrays")
        let addFunctionPSO = try? device?.makeComputePipelineState(function: addFunction!)
        let commandQueue = device?.makeCommandQueue()
        let bufferDataA = (0..<bufferLength).map { _ in randomFloat() }
        let bufferDataB = (0..<bufferLength).map { _ in randomFloat() }
        let bufferA = device?.makeBuffer(bytes: bufferDataA, length: bufferSize, options: .storageModeShared)
        let bufferB = device?.makeBuffer(bytes: bufferDataB, length: bufferSize, options: .storageModeShared)
        
        let bufferResult = device?.makeBuffer(length: bufferSize, options: .storageModeShared)
        let commandBuffer = commandQueue?.makeCommandBuffer()
        let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
        computeEncoder?.setComputePipelineState(addFunctionPSO!)
        computeEncoder?.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder?.setBuffer(bufferB, offset: 0, index: 1)
        computeEncoder?.setBuffer(bufferResult, offset: 0, index: 2)

        let maxTotalThreadsPerThreadgroup = addFunctionPSO!.maxTotalThreadsPerThreadgroup
        let threadGroupCount = MTLSizeMake(min(maxTotalThreadsPerThreadgroup, bufferLength), 1, 1)
        let threadGroups = MTLSizeMake(bufferLength / threadGroupCount.width, 1, 1)
        computeEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
        computeEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        let floatBufferPointer = UnsafeBufferPointer(start: bufferResult?.contents().assumingMemoryBound(to: Float.self), count: bufferLength)
        let result = [Float](floatBufferPointer)
        var error = false
        for index in 0..<bufferLength {
            if (abs(bufferDataA[index] + bufferDataB[index] - result[index]) > Float.ulpOfOne) {
                print("Error")
                error = true
            }
        }
        if (!error) {
            print("Ok")
        }
    }

    func tryCompile(_ device: MTLDevice?, _ function: MTLFunction?) -> MTLComputePipelineState? {
        do {
            let computePipelineState = try device?.makeComputePipelineState(function: function!)
            return computePipelineState
        } catch let error {
            print(error.localizedDescription)
            return nil
        }
    }

    @IBAction func onReduce(_ sender: Any) {
        let device = MTLCreateSystemDefaultDevice()
        if device?.supportsFamily(.apple7) == false {
            print("GPU family less than Apple7 detected - this probably won't work")
        }
        let defaultLibrary = device?.makeDefaultLibrary()
        let reduceFunction = defaultLibrary?.makeFunction(name: "reduce")
        guard let reduceFunctionPSO = tryCompile(device, reduceFunction) else {
            return
        }
        let commandQueue = device?.makeCommandQueue()
        let bufferDataA = (0..<bufferLength).map { _ in randomInt32SmallPositive() }
        let bufferA = device?.makeBuffer(bytes: bufferDataA, length: bufferSize, options: .storageModeShared)

        let bufferResult = device?.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared)
        let commandBuffer = commandQueue?.makeCommandBuffer()
        let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
        computeEncoder?.setComputePipelineState(reduceFunctionPSO)
        computeEncoder?.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder?.setBuffer(bufferResult, offset: 0, index: 1)
        let maxTotalThreadsPerThreadgroup = reduceFunctionPSO.maxTotalThreadsPerThreadgroup
        let threadGroupCount = MTLSizeMake(min(maxTotalThreadsPerThreadgroup, bufferLength), 1, 1)
        let threadGroups = MTLSizeMake(bufferLength / threadGroupCount.width, 1, 1)
        computeEncoder?.setThreadgroupMemoryLength(threadGroupCount.width, index: 0)
        computeEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
        computeEncoder?.endEncoding()
        commandBuffer?.commit()

        commandBuffer?.waitUntilCompleted()
        let intBufferPointer = UnsafeBufferPointer(start: bufferResult?.contents().assumingMemoryBound(to: Int32.self), count: 1)
        let result = [Int32](intBufferPointer)
        var sum:Int32 = 0
        for index in 0..<bufferLength {
            sum += bufferDataA[index]
        }
        print(sum == result[0] ? "Ok" : "Error")
    }
    
    var bitonicFunctionPSO: MTLComputePipelineState?
    var bitonicCommandQueue: MTLCommandQueue?
    var bitonicDataBuffer: MTLBuffer?
    var bitonicParameterBuffer: MTLBuffer?
    var commandBuffer: MTLCommandBuffer?
    var computeEncoder: MTLComputeCommandEncoder?
    var runCount: UInt32 = 0

    func calcTotalRunCount(_ maxWorkgroupSize: UInt32) -> UInt32 {
        let n = UInt32(bufferLength)
        let workgroupSizeX = (n < maxWorkgroupSize * 2) ? (n / 2) : maxWorkgroupSize

        let outerRunCount: UInt32 = { (N: UInt32) -> UInt32 in var ret: UInt32 = 0; var n = n; while (n > 1) { n >>= 1; ret = ret + 1; }; return ret } (n / workgroupSizeX)
        return outerRunCount * (outerRunCount + 1) / 2;
    }

    @IBAction func onBitonic(_ sender: Any) {
        let device = MTLCreateSystemDefaultDevice()
        let defaultLibrary = device?.makeDefaultLibrary()
        let bitonicFunction = defaultLibrary?.makeFunction(name: "bitonic")
        bitonicFunctionPSO = try? device?.makeComputePipelineState(function: bitonicFunction!)
        let maxWorkgroupSize = UInt32(bitonicFunctionPSO!.maxTotalThreadsPerThreadgroup)
        let totalRunCount = calcTotalRunCount(maxWorkgroupSize)
        bitonicCommandQueue = device?.makeCommandQueue()
        let bufferData = (0..<bufferLength).map { _ in randomInt32() }
        bitonicDataBuffer = device?.makeBuffer(bytes: bufferData, length: bufferSize, options: .storageModeShared)

        bitonicParameterBuffer = device!.makeBuffer(length: MemoryLayout<Parameters>.size * Int(totalRunCount))
        
        let n = UInt32(bufferLength)
        let workgroupSizeX = (n < maxWorkgroupSize * 2) ? (n / 2) : maxWorkgroupSize

        let workgroupCount = n / (workgroupSizeX * 2 )

        var h:UInt32 = workgroupSizeX * 2
        assert(h <= n)
        assert(h % 2 == 0)

        commandBuffer = bitonicCommandQueue?.makeCommandBuffer()
        computeEncoder = commandBuffer?.makeComputeCommandEncoder()

        localBitonicMergeSort(h, workgroupCount)
        // we must now double h, as this happens before every flip
        h *= 2

        while (h <= n) {
            bigFlip(h, workgroupCount);
            var hh = h / 2
            while (hh > 1) {
                if (hh <= workgroupSizeX * 2) {
                    // We can fit all elements for a disperse operation into continuous shader
                    // workgroup local memory, which means we can complete the rest of the
                    // cascade using a single shader invocation.
                    localDisperse(hh, workgroupCount)
                    break
                } else {
                    bigDisperse(hh, workgroupCount)
                }
                hh /= 2
            }
            h *= 2
        }
        computeEncoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()

        let intBufferPointer = UnsafeBufferPointer(start: bitonicDataBuffer?.contents().assumingMemoryBound(to: Int32.self), count: bufferLength)
        let result = [Int32](intBufferPointer)
        var error = false
        for index in 0..<bufferLength-1 {
            if result[index] < result[index + 1] {
                print("Error")
                error = true
            }
        }
        if !error {
            print("Ok")
        }
    }

    func localBitonicMergeSort(_ h: UInt32, _ workgroupCount: UInt32) {
        dispatch(h, eLocalBitonicMergeSort, workgroupCount)
    }

    func bigFlip(_ h: UInt32, _ workgroupCount: UInt32) {
        dispatch(h, eBigFlip, workgroupCount)
    }

    func localDisperse(_ h: UInt32, _ workgroupCount: UInt32) {
        dispatch(h, eLocalDisperse, workgroupCount)
    }

    func bigDisperse(_ h: UInt32, _ workgroupCount: UInt32) {
        dispatch(h, eBigDisperse, workgroupCount)
    }

    func dispatch(_ h: UInt32, _ algorithm: eAlgorithmVariant,  _ workgroupCount: UInt32) {
        let bitonicParameter = Parameters(h: h, algorithm: algorithm)
        bitonicParameterBuffer?.contents().storeBytes(of: bitonicParameter, toByteOffset: Int(runCount) * MemoryLayout<Parameters>.size, as: Parameters.self)
        computeEncoder?.setComputePipelineState(bitonicFunctionPSO!)
        computeEncoder?.setBuffer(bitonicDataBuffer, offset: 0, index: 0)
        computeEncoder!.setBuffer(bitonicParameterBuffer, offset: Int(runCount) * MemoryLayout<Parameters>.size, index: 1)
        let workGroupSize = bitonicFunctionPSO!.maxTotalThreadsPerThreadgroup
        let threadgroupMemoryLength = workGroupSize * 2
        computeEncoder!.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)

        let threadGroupCount = MTLSizeMake(workGroupSize, 1, 1)
        let threadGroups = MTLSizeMake(Int(workgroupCount), 1, 1)
        computeEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
        runCount = runCount + 1
    }

    @IBOutlet weak var addButton: UIButton!
    @IBOutlet weak var reduceButton: UIButton!
    @IBOutlet weak var bitonicButton: UIButton!
}

