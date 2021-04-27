import os, time
import numpy as np
import numba as nb
import cv2
import ctypes
#import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class YoloDetector():
    def __init__(self, img_size, config, logger):
        self.img_size = img_size
        self.TRT_LOGGER = logger
        self.config = config
        self.batch_size = 1
        self.max_batch_size =  config['max_batch_size']
        self.engine_path = config['engine_path']
        self.trt_runtime = trt.Runtime(self.TRT_LOGGER)
        # print(os.listdir('models'))
        if os.path.exists(self.engine_path):
          with open(self.engine_path, "rb") as f:
              self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())
          self.context = self.engine.create_execution_context()
          print("max_batch_size", self.max_batch_size)
          self.context.set_binding_shape(0, (self.max_batch_size, 3, self.img_size, self.img_size))
          self.inputs = []
          self.outputs = []
          self.bindings = []
          self.stream = cuda.Stream()
          for binding in self.engine:
              size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
              dims = self.engine.get_binding_shape(binding)        
              if dims[0] < 0: size *= -1        
              dtype = trt.nptype(self.engine.get_binding_dtype(binding))
              # Allocate host and device buffers
              print("size, dtype", size, dtype, dims)
              host_mem = cuda.pagelocked_empty(size, dtype)
              device_mem = cuda.mem_alloc(host_mem.nbytes)
              # Append the device buffer to device bindings.
              self.bindings.append(int(device_mem))
              # Append to the appropriate list.
              if self.engine.binding_is_input(binding): 
                  self.inputs.append(HostDeviceMem(host_mem, device_mem))
              else: 
                  self.outputs.append(HostDeviceMem(host_mem, device_mem))
        else:
          print("!!!!!!!!!!!!Engine does not exist!!", self.engine_path)


    def detect_async(self, frames):
        frames_tf = np.transpose(frames, (2, 3, 1, 0)).astype(np.float32) / 255.0
        frames_tf = np.stack(frames, axis=0)
        frames_tf = np.ascontiguousarray(frames_tf)
        self.batch_size = len(frames)
        self.inputs[0].host = frames_tf
        self.do_inference_new(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=self.batch_size)


    def postprocess(self):
        self.stream.synchronize()
        trt_outputs = [out.host for out in self.outputs]
        print("type outputs", trt_outputs)
        # trt_outputs = trt_outputs.cpu().detach().numpy()
        boxes = self.post_processing(0, 0.3, 0.4, self.img_size, trt_outputs, self.batch_size)
        return boxes

    def do_inference_new(self, context, bindings, inputs, outputs, stream, batch_size=1):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        #self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # https://github.com/numba/numba/issues/2411
    def post_processing(self, cls, conf_thresh, nms_thresh, scale, output, batch_size):
        box_array = output[0].reshape(batch_size, -1, 1, 4)
        confs = output[1].reshape(batch_size, -1, 80)
        box_array = box_array[:, :, 0]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_max_id = max_id[i, argwhere]
            bboxes = []
            j = cls
            if cls in l_max_id:
                l_box_array = box_array[i, argwhere, :]
                l_max_conf = max_conf[i, argwhere]
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self._nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append((ll_box_array[k])) * scale 
            bboxes_batch.append(bboxes)
        return bboxes_batch

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _nms_cpu(boxes, confs, nms_thresh):
        # print(boxes.shape)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        return np.array(keep)

    # @staticmethod
    # @nb.njit(fastmath=True, cache=True)
    # def _preprocess(frames):
    #     frames_tf = np.transpose(frames, (2, 3, 1, 0)).astype(np.float32) / 255.0
    #     frames_tf = np.stack(frames, axis=0)
    #     frames_tf = np.ascontiguousarray(frames_tf)
    #     return frames_tf