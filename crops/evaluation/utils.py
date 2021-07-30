import torch

def nonmaximalsuppression(tensor, threshold):
    pred_data = tensor.storage()
    offset = tensor.storage_offset()
    stride = int(tensor.stride()[0])
    numel = tensor.numel()
    points = []

    # Corners
    val = pred_data[0 + offset]
    if val >= threshold and val >= pred_data[1 + offset] and val >= pred_data[stride + offset]:
        points.append([0, 0])

    val = pred_data[stride - 1 + offset]
    if val >= threshold and val >= pred_data[stride - 2 + offset] and val >= pred_data[2 * stride - 1 + offset]:
        points.append([stride - 1, 0])
        
    val = pred_data[numel - stride + offset]
    if val > threshold and val >= pred_data[numel - stride + 1 + offset] and val >= pred_data[numel - 2 * stride + offset]:
        points.append([0, stride - 1])

    val = pred_data[numel - 1 + offset]
    if val > threshold and val >= pred_data[numel -2 + offset] and val >= pred_data[numel - 1 - stride + offset]:
        points.append([stride - 1, stride - 1])

    # Top y==0
    for i in range(1,stride-1):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i-1] and val >= pred_data[i+1] and val >= pred_data[i+stride]:
            points.append([i - offset, 0])

    # Bottom y==stride-1
    for i in range(numel-stride+1,numel-1):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i-1] and val >= pred_data[i+1] and val >= pred_data[i-stride]:
            points.append([i - numel + stride - offset, stride])

    # Front x==0
    for i in range(stride, stride * (stride - 1), stride):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i+stride] and val >= pred_data[i-stride] and val >= pred_data[i+1]:
            points.append([0, (i - offset) // stride])

    # Back x == stride-1
    for i in range(stride - 1, stride * (stride - 1), stride):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i+stride] and val >= pred_data[i-stride] and val >= pred_data[i-1]:
            points.append([stride - 1, (i - offset) // stride])

    # Remaining inner pixels
    for i in range(stride+1, stride * (stride - 1), stride):
        for j in range(i,i+stride-2):
            j += offset
            val = pred_data[j]
            if val >= threshold and val >= pred_data[j+1] and val >= pred_data[j-1] and val >= pred_data[j+stride] and val >= pred_data[j-stride]:
                points.append([(j - offset) % stride, i // stride])

    return points