import glob, os

from attr import Attribute

from create_annotations import *
import cv2

category_ids = {
    "chicken": 0
}

category_colors = {
    "255": 0 
}

multipolygon_ids = [0]

def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for mask_image in glob.glob(maskpath + "*.jpg"):
        # mask_image = "Dataset/train/mask/9_4376_20.jpg"
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("L")
        mask_image_open = mask_image_open.point( lambda p: 255 if p > 200 else 0 )
        mask_image_open = mask_image_open.convert("1")
        # print(np.array(mask_image_open))
        # img_cv2 = cv2.cvtColor(np.array(mask_image_open), cv2.COLOR_RGB2BGR)
        # cv2_imshow(img_cv2)
        w, h = mask_image_open.size
        
        # "images" info 
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
    
        # set color equal to chicken color mask
        color = '255'
        sub_mask = sub_masks[color]
        category_id = category_colors[color]

        # "annotations" info
        
        polygons, segmentations = create_sub_mask_annotation(sub_mask)
        # print(polygons)
        
        # Check if we have classes that are a multipolygon
        if category_id in multipolygon_ids:
            # Combine the polygons to calculate the bounding box and area
            
            multi_poly = MultiPolygon(polygons)
                            
            annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
            

            annotations.append(annotation)
            annotation_id += 1
        else:
            for i in range(len(polygons)):
                # Cleaner to recalculate this variable
                segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                
                annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                
                annotations.append(annotation)
                annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id

coco_format = get_coco_json_format()
    
keyword = "train"
mask_path = "Dataset/{}/mask/".format(keyword)

# Create category section
coco_format["categories"] = create_category_annotation(category_ids)

# Create images and annotations sections
coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

with open("{}.json".format(keyword),"w") as outfile:
    json.dump(coco_format, outfile, indent=4)

print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

