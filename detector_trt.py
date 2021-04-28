import sys
import os
import time
import argparse
import numpy as np
import numba as nb
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class YoloDetector():
  def __init__(self, config, logger):
    self.TRT_LOGGER = logger
    self.config = config
    print(self.config)
    self.img_size = self.config['img_size']
    self.max_batch_size = self.config['max_batch_size']
    self.batch_size = 1
    self.image_size = (self.img_size, self.img_size)
    with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
        self.engine = runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()
    self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.max_batch_size)
    self.context.set_binding_shape(0, (self.max_batch_size, 3, self.img_size, self.img_size))

  def detect(self, frames):
      self.batch_size = len(frames)
      img_in = np.transpose(frames, (0, 3, 1, 2)).astype(np.float32)/255.0
      img_in = np.stack(img_in, axis=0)
      img_in = np.ascontiguousarray(img_in)
      t1 = time.perf_counter()
      self.inputs[0].host = img_in
      self.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

  # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
  def allocate_buffers(self, engine, batch_size):
      inputs = []
      outputs = []
      bindings = []
      stream = cuda.Stream()
      for binding in engine:
          size = trt.volume(engine.get_binding_shape(binding)) * batch_size
          dims = engine.get_binding_shape(binding)        
          if dims[0] < 0: size *= -1        
          dtype = trt.nptype(engine.get_binding_dtype(binding))
          host_mem = cuda.pagelocked_empty(size, dtype)
          device_mem = cuda.mem_alloc(host_mem.nbytes)
          bindings.append(int(device_mem))
          if engine.binding_is_input(binding):
              inputs.append(HostDeviceMem(host_mem, device_mem))
          else:
              outputs.append(HostDeviceMem(host_mem, device_mem))
      return inputs, outputs, bindings, stream

  def do_inference(self, context, bindings, inputs, outputs, stream):
      [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
      context.execute_async(bindings=bindings, stream_handle=stream.handle)
      [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

  def postprocess(self):
      self.stream.synchronize()
      trt_outputs = [out.host for out in self.outputs]     
      boxes = self.post_processing(0.4, 0.6, trt_outputs, self.img_size, self.batch_size, 0)
      return boxes

  # @nb.njit(fastmath=True, cache=True)
  def post_processing(self, conf_thresh, nms_thresh, output, scale, batch_size, cls):
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
            if np.in1d(cls, l_max_id):
                l_box_array = box_array[i, argwhere, :]
                l_max_conf = max_conf[i, argwhere]
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self.nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(ll_box_array[k] * scale)  
            bboxes_batch.append(bboxes)
        return bboxes_batch
  # @staticmethod
  # @nb.njit(fastmath=True, cache=True) # 8 detected 3 0.035231882000516634 0.00947332600117079 0.025759570999071002
  #                                       8 detected 3 0.03639759900033823 0.009599693003110588 0.026798553997650743
  def nms_cpu(self, boxes, confs, nms_thresh=0.5): 
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

if __name__ == '__main__':
    engine_path = sys.argv[1]
    img_size = 416
    config_mot = {'engine_path':'models/2_yolov4_1_3_416_416_dynamic.engine','max_batch_size':4,'img_size':img_size}

    image_size = (img_size, img_size)    
    image_path_0 = 'video/39_2.jpg' 
    image_src_0 = cv2.imread(image_path_0)
    image_src_0 = cv2.cvtColor(image_src_0, cv2.COLOR_BGR2RGB)
    image_src_0 = cv2.resize(image_src_0, image_size, interpolation=cv2.INTER_LINEAR)

    image_path_1 = 'video/39_2.jpg' 
    image_src_1 = cv2.imread(image_path_1)
    image_src_1 = cv2.cvtColor(image_src_1, cv2.COLOR_BGR2RGB)
    image_src_1 = cv2.resize(image_src_1, image_size, interpolation=cv2.INTER_LINEAR)
    frames = [image_src_1, image_src_0, image_src_1]
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print("init detect")
    # main(engine_path, frames)
    detector = YoloDetector(config_mot, TRT_LOGGER)
    print("init ok")
    if frames:
      print("start detect")
      for i in range(10):
        t1 = time.perf_counter()
        detector.detect(frames)
        t2 = time.perf_counter()
        boxes = detector.postprocess()
        print(i, 'detected', len(boxes), time.perf_counter()-t1, t2-t1, time.perf_counter()-t2)
      for i in range(len(boxes)):
        for box in boxes[i]:
          tlbr = box.astype(int)
          tl, br = tuple(tlbr[:2]), tuple(tlbr[2:])
          cv2.rectangle(frames[i], tl, br, (255,0,0), 2)  
        cv2.imwrite('img_trt_det_'+str(i)+'.jpg', frames[i])
        print("saved file",'img_trt_det_'+str(i)+'.jpg')
    ctx.pop()