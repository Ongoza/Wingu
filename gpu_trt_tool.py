import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
#import time

def post_processing(cls, conf_thresh, nms_thresh, output, img_size):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    #t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()
    #num_classes = confs.shape[2]
    # [batch, num, 4]
    box_array = box_array[:, :, 0]
    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    #t2 = time.time()
    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_max_id = max_id[i, argwhere]
        # print("l_max_id",l_max_id)
        bboxes = []
        # nms for each class
        j = cls
        if cls in l_max_id:
            # for j in range(num_classes):
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]
            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :] * img_size
                #ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]
                if ll_max_id == cls:
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3]])
        bboxes_batch.append(bboxes)
    #t3 = time.time()
    #print('-----------------------------------')
    #print('       max and argmax : %f' % (t2 - t1))
    #print('                  nms : %f' % (t3 - t2))
    #print('Post processing total : %f' % (t3 - t1))
    #print('-----------------------------------')
    return bboxes_batch

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
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
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def GiB(val):
    return val * 1 << 30

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0: size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def get_engine(engine_path, TRT_LOGGER):
    # If a serialized engine exists, use it instead of building an engine.
    #print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def detect(context, buffers, images, buf_size=1, img_size=416):
    #ta = time.time()
    inputs, outputs, bindings, stream = buffers
    #print('Length of inputs: ', len(inputs))
    inputs[0].host = images
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #print('Len of outputs: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
    trt_outputs[0] = trt_outputs[0].reshape(buf_size, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(buf_size, -1, 80)
    boxes = post_processing(0, 0.3, 0.3, trt_outputs, img_size)
    #print('Len of outputs 2: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
    #tb = time.time()
    #print('TRT inference time: %f' % (time.time() - ta))
    return boxes