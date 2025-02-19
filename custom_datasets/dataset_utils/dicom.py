from pydicom import dcmread

def get_pixel_array(dcm_file):
    '''
    Convert DICOM pixel array into image format.
    Adapted from https://stackoverflow.com/questions/42650233/how-to-access-rgb-pixel-arrays-from-dicom-files-using-pydicom
    '''
    dcm = dcmread(dcm_file)
    img = dcm.pixel_array

    # Rescale pixel array
    if hasattr(dcm, 'RescaleSlope'):
        img = img * dcm.RescaleSlope
    if hasattr(dcm, 'RescaleIntercept'):
        img = img + dcm.RescaleIntercept

    # Convert pixel_array (img) to -> gray image (img_2d_scaled)
    ## Step 1. Convert to float to avoid overflow or underflow losses.
    img_2d = img.astype(float)

    ## Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    ## Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)

    ## Step 4. Invert pixels if MONOCHROME1
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img_2d_scaled = np.invert(img_2d_scaled)

    return img_2d_scaled