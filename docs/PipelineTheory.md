input: seperate RFP and DAPI flourecent image pairs ( reffered to as red_img and blue_img)

pipeline:
    (red_image, blue_img) -> 
    grayscale ->  
    extractor_run: ->
        red_img & blue_img -> %%it looks like i dont normalize them%%
        prep grayscale masks: ->
            mask_r_dilation = np.maximum(blue, red) 
            mask_r_erosion = np.minimum(blue, red)
            %%these are np/ basicaly images of where each pixel value comes from whichever wins there%%
        diff_r = bool where red stronger than min envelope: ->
            diff_r = red > mask_r_erosion
        clean up diff_r: ->
            binary erosion with a 3x3 square kernel ->
            remove small objects less then 100 px area ->
        
        abs_diff = Secondary mask using absolute difference: -> 
            abs of mask_r_dilation - red
            mask_gt_red = abs_diff > red (this is a bool)
        then create the eroded mask

        


        
        get: (background_tissue_intensity, average_crypt_intensity)



    note: cant take combo straigt from imager as it preforms some sort of preproc that significantly changes the images, likely they are normalized on combined scales instead of seperate scales
