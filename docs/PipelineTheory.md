input: seperate RFP and DAPI flourecent image pairs ( reffered to as red_img and blue_img)

pipeline:
    (red_image, blue_img) -> 
    grayscale ->  
    extractor_run: ->
        red_img & blue_img -> %%it looks like i dont normalize them%% 
            # in general, Dapi has overal high signals for anywhere that has tissue
            ## and RFP has lower signals overall for tissue, with slightly higher signals for the crypts
        %%prep grayscale masks: ->
            mask_r_dilation = np.maximum(blue, red) # this is just a cap across the image, for the max values between the two
            mask_r_erosion = np.minimum(blue, red) # this creates an image where all of the noncrypt tissue is dropped in intesity, while the cypts remain high
            these are np/ basicaly images of where each pixel value comes from whichever wins there  
            purged for redundancy %%

        diff_r = bool where red stronger than min envelope: ->
            diff_r = red > blue # aka mask of crypts are

            clean up diff_r: ->
                binary erosion with a 3x3 square kernel ->
                remove small objects less then 100 px area ->
            # 
        
        Secondary mask using absolute difference = mask of non crypt tissue -> 
            mask_gt_red = blue > 2*red = non-crypt-tissue
            mask_gt_red_eroded = mask_gt_red is realy speckeled by noise, so this fixes it

        label handeling: ->
            Combined Labels = (0 bg, 1 diff_r, 2 mask_gt_red_eroded) aka ( 0 bg, 1 cypts, 2 non crypt tissue)
            then the diff_r & mask_gt_red_eroded get expanded by 100px to meet/ fill up bg between them->
            use ndi_label to seperatly label disconnected regions of diff_r-> ( this effectivly leaes )

        set up reworked labels ( which are used as the watershed markers) -> 
            1 =expanded_labels[2] =  mask_gt_red_eroded = non-crypt-tissue
            mask_copy = select pixels that are NOT in expanded class 2 AND that belong to a connected component from diff_r.
                i.e., crypt pixels that lie outside the class-2 region (or in class-1/background area).
            reworked[mask_copy] = copy the connected-component ids into the markers image, shifting by +1 so they don't collide with the marker 1 used for class 2.
                result: each diff_r component gets its own marker label (2,3,4,...).
            %%this basicaly allows us to get much tigher and precise marks for each of the crypts so that we start with much more confdent positions, and prevent the overmergeing seen in raw expanded labels, while still having the very intact non crypt tissue regions of expanded labels %%

        then we get elevation: -> 
            this is a distance transform on combined lables, where non crypt tissue mountains, and crypt tissue form valies
         
        then using the reworked lables as markers and combined label derived elevation we run watershed
            getting our likely crypt candidates

    scoring:
        uses a set of self descripteve weights:
            self.weights = weights if weights is not None else {
                'circularity': 0.35,    # Most important - want circular regions
                'area': 0.25,           # Second - want consistent sizes
                'line_fit': 0.15,       # Moderate - want aligned regions
                'red_intensity': 0.15,  # Moderate - want bright regions
                'com_consistency': 0.10 # Least - center consistency
            }
        line fit reffers to proximity to a line through the center of mass of all the detected crypt region, this is an approximation of "crypts are genraly arrayed in a straightish line along the gut wall
        by default choses the 5 best
        
        get: (background_tissue_intensity, average_crypt_intensity)

ok, so theres the pipeline, but how does it work:

    starting with the RFP(red) and DAPI(Blue) images, we first find the 




    note: cant take combo straigt from imager as it preforms some sort of preproc that significantly changes the images, likely they are normalized on combined scales instead of seperate scales

normalization across images:
    since for a given slide the contrast between the crypt and non crypt staining/signal can varry, and since we want consitant and meaningfull crypt RFP intensities, we need to crypts of each image in context of their overal RFP and dapi affinity, this is done by finding the ratio of RFP to DAPI of the crypt and non crypt tissue seperatly and using the ratio between those two to standardize across images. this effectivly sets a standardized gain like factor.