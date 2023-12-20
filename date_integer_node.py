from datetime import datetime
from pickle import APPEND
import os
import torch
import numpy as np
import json
import textwrap
from PIL import Image, ImageOps, ImageDraw, ImageFont
import folder_paths

def process_text(anything):
    return {"text": [str(anything)]}

class MetadataOverlayNode:
    CATEGORY = "RenderRiftNodes"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_text_on_image"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image1": ("IMAGE",),
                "metadata1": ("METADATA",),
                "image2": ("IMAGE",),
                "metadata2": ("METADATA",),
                "checkpoint": ("BOOLEAN", {"default": False}),
                "ksampler": ("BOOLEAN", {"default": False}),
                "controlnets": ("BOOLEAN", {"default": False}),
                "animdiff": ("BOOLEAN", {"default": False}),
                "ipadpter": ("BOOLEAN", {"default": False}),
                "loras": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "original_vid_optional": ("IMAGE",),
                "image3": ("IMAGE",),
                "metadata3": ("METADATA",),
                "image4": ("IMAGE",),
                "metadata4": ("METADATA",),
                "image5": ("IMAGE",),
                "metadata5": ("METADATA",),
                "image6": ("IMAGE",),
                "metadata6": ("METADATA",),
            },
        }

    def __init__(self):
        pass

    
    def extract_metadata(self, metadata, checkpoint, ksampler, controlnets, animdiff, ipadpter, loras):
        if metadata == "Original Video":
            return "Original Video"

        metadata = metadata.get('prompt')
        if metadata:
            try:
                # Parse the JSON metadata
                json_data = json.loads(metadata)
                prompt_data = json_data.get("prompt", json_data)

                # Dictionary to hold class_type and their details
                class_type_details = {}

                # Process prompt_data to extract and organize specific attributes
                for node_id, node in prompt_data.items():
                    node_type = node.get('class_type')
                    inputs = node.get('inputs', {})
                    # Collect the details for each class_type of interest
                    if animdiff and node_type == "ADE_AnimateDiffLoaderWithContext":
                        # Extract the context_options ID from the inputs
                        context_options_id = inputs.get('context_options', [])[0]
                        context_details = ""
                        # If context_options ID is present, find the related node and extract details
                        if context_options_id:
                            context_node = prompt_data.get(context_options_id, {})
                            context_inputs = context_node.get('inputs', {})
                            context_details = f" context_length: {context_inputs.get('context_length')}, context_stride: {context_inputs.get('context_stride')}, context_overlap: {context_inputs.get('context_overlap')}, context_schedule: {context_inputs.get('context_schedule')}, closed_loop: {context_inputs.get('closed_loop')}"
        
                        details = f"AnimateDiff Details \n ade_model: {inputs.get('model_name')} \n motion_scale: {inputs.get('motion_scale')} {context_details}"
                        class_type_details.setdefault(node_type, []).append(details)
                    # Latent or upscale, IPAdapter, IPAdapter Batch Load, Animif diff, Loras, ControlNets, KSampler
                    elif controlnets and node_type == "ControlNetLoaderAdvanced":
                        apply_node_id = next((nid for nid, n in prompt_data.items() if n.get('class_type') == "ControlNetApplyAdvanced" and n['inputs'].get('control_net', [])[0] == node_id), None)
                        if apply_node_id:
                            apply_inputs = prompt_data[apply_node_id]['inputs']
                            details = f"CN: {inputs.get('control_net_name')} \n strength: {apply_inputs.get('strength')} \n start_per: {apply_inputs.get('start_percent')} \n end_per: {apply_inputs.get('end_percent')}"
                            class_type_details.setdefault(node_type, []).append(details)
                    elif checkpoint and node_type == "CheckpointLoaderSimple":
                        details = f"ckpt: {inputs.get('ckpt_name')}"
                        class_type_details.setdefault(node_type, []).append(details)
                    elif ksampler and node_type == "KSamplerAdvanced":
                        details = f"Ksamples - steps: {inputs.get('steps')}, cfg: {inputs.get('cfg')} \n sampler: {inputs.get('sampler_name') + ' ' + inputs.get('scheduler')}"
                        class_type_details.setdefault(node_type, []).append(details)
                    elif loras and node_type == "LoraLoader":
                        details = f"lora_name: {inputs.get('model')}, strength_model: {inputs.get('strength_model')}"
                        class_type_details.setdefault(node_type, []).append(details)
                # Prepare the output string
                    output = '\n'.join(['\n'.join(details) for details in class_type_details.values()])

                return output

            except json.JSONDecodeError:
                return "Error parsing metadata. Ensure it is in valid JSON format."
        else:
            return "Metadata not found in file."



    def overlay_text_on_image(
        self, 
        image1,
        metadata1,
        image2,
        metadata2,
        checkpoint=False,
        ksampler=False,
        controlnets=False,
        animdiff=False,
        ipadpter=False,
        loras=False,
        original_vid_optional=None,
        image3=None,
        metadata3=None,
        image4=None,
        metadata4=None,
        image5=None,
        metadata5=None,
        image6=None,
        metadata6=None,
        ):
       
        images_and_metadata = []
        if original_vid_optional is not None:
            images_and_metadata.append((original_vid_optional, "Original Video"))


        # Extend the list with other image and metadata pairs
        additional_images_and_metadata = [
            (image1, metadata1),
            (image2, metadata2),
            (image3, metadata3),
            (image4, metadata4),
            (image5, metadata5),
            (image6, metadata6),
        ]

        # Filter out None values and add to the main list
        images_and_metadata.extend([(img, meta) for img, meta in additional_images_and_metadata if img is not None and meta is not None])


        # List to store overlay images
        overlay_images = []
            # Extract all metadata texts and calculate the maximum border height
    #   Find the width of the largest image
        if image1 is not None:
            max_image_width = image1.shape[2]  # Assuming image1 is in CxHxW format
            max_text_width = max_image_width - 20  # Consistent text width


        all_texts = [self.extract_metadata(meta, checkpoint, ksampler, controlnets, animdiff, ipadpter, loras) for _, meta in images_and_metadata]
        max_border_height = self.calculate_max_border_height(all_texts, max_text_width)

        # Now process each image with the standardized border height
        for img, meta in images_and_metadata:
            text = self.extract_metadata(meta, checkpoint, ksampler, controlnets, animdiff, ipadpter, loras)
            overlay_img = self.overlayText(img, text, max_border_height, max_image_width )
            overlay_images.append(overlay_img[0] if isinstance(overlay_img, tuple) else overlay_img)


        num_images = len(overlay_images)
        if num_images % 3 != 0 and num_images > 2:
            # Calculate how many blank images are needed to complete the row
            num_blank_images = 3 - (num_images % 3)

            # Create and append blank images
            blank_image_size = overlay_images[0].shape
            for _ in range(num_blank_images):
                overlay_images.append(self.create_blank_image(blank_image_size))

        # Concatenate images into rows with a maximum of 3 images per row
        rows = []
        for i in range(0, len(overlay_images), 3):
            # Ensure consistent dimensions
            images_to_concat = [img.squeeze(0) if img.dim() == 4 else img for img in overlay_images[i:i+3]]
            rows.append(torch.cat(images_to_concat, dim=2))

        # Concatenate rows vertically
        final_image = torch.cat(rows, dim=1) if rows else torch.empty(0)

        return (final_image,)


    def create_blank_image(self, size):
        # Create a blank image tensor of the specified size
        # Assuming the images are 3-channel (RGB)
        return torch.zeros(size)

    def calculate_max_border_height(self, texts, max_text_width, font_size=40 ):
        font = ImageFont.truetype("arial.ttf", 20)
        max_border_height = 100  # Starting with a minimum height

        for text in texts:
            # Calculate required height for this text
            wrapped_text = self.wrap_text(text, font, max_text_width)  # Assuming max_width is set
            line_height = font.getsize('A')[1] + 10
            total_text_height = len(wrapped_text) * line_height
            border_height = max(100, total_text_height + 20)
            max_border_height = max(max_border_height, border_height)

        return max_border_height
    def wrap_text(self, text, font, max_width):
        lines = []
        
        # Split the text by newlines to preserve intentional line breaks
        paragraphs = text.split('\n')

        for paragraph in paragraphs:
            words = paragraph.split()
            if not words:  # This paragraph was an intentional line break
                lines.append('')
                continue
            line = ''
            for word in words:
                # Check if the word can be added to the line
                if font.getsize(line + word)[0] <= max_width:
                    line += word + ' '
                else:
                    # If the line is not empty, append it and start a new line
                    if line:
                        lines.append(line)
                        line = ''
                    # If the word itself exceeds the max width, it's placed on a new line
                    line = word + ' '
            # Append the last line of the paragraph
            lines.append(line)

        return [line.rstrip() for line in lines]
    def overlayText(self, image, text, max_border_height, max_text_width):
        processed_images = []

        for image_tensor in image:
            image_np = image_tensor.cpu().numpy()
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))

            # Settings for font and text
            font = ImageFont.truetype("arial.ttf", 20)
            max_width = max_text_width 

            # Use a function to wrap text based on pixel width

            # Get the wrapped text
            wrapped_text = self.wrap_text(text, font, max_width)

            # Determine the height of each line (including padding)
            line_height = font.getsize('A')[1] + 10  # Example height of a single line of text with padding

            # Calculate total height required for wrapped text
            # total_text_height = len(wrapped_text) * line_height

            # Adjust border_height to fit all text
            border_height = max_border_height

            # Create a new image with extra space for the border
            new_image = Image.new("RGB", (pil_image.width, pil_image.height + border_height), "black")
            new_image.paste(pil_image, (0, border_height))

            # Draw text on the border
            draw = ImageDraw.Draw(new_image)
            y_text = 10  # Starting position for text

            for line in wrapped_text:
                draw.text((10, y_text), line, fill=(255, 255, 255), font=font)
                y_text += line_height  # Move down the y position for the next line

            # Convert back to Tensor
            image_tensor_out = torch.tensor(np.array(new_image).astype(np.float32) / 255.0)
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            processed_images.append(image_tensor_out)

        # Stack processed images along the batch dimension
        final_tensor = torch.cat(processed_images, dim=0)
        return final_tensor



    def tensor_to_pil(self, image_tensor):
        image_np = image_tensor.cpu().numpy()
        return Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))

    def resize_image(self, image, height):
        width = int(image.width * height / image.height)
        return image.resize((width, height), Image.ANTIALIAS)

    def add_text_to_image(self, image, text):
        # Create a new image with extra space for the border
        border_height = 100
        new_image = Image.new("RGB", (image.width, image.height + border_height), "black")
        new_image.paste(image, (0, border_height))

        # Draw text on the border
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("arial.ttf", 40)
        draw.text((10, (border_height - 40) // 2), text, fill=(255, 255, 255), font=font)

        return new_image
 
class TextTokens:
    def __init__(self):
        # Initialize any necessary variables or settings
        pass

    def parseTokens(self, text):
        # For simplicity, this method currently does nothing and just returns the text.
        # You can add token parsing logic here if needed.
        return text

class DateIntegerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False}),
                "int_1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "text_c": ("STRING", {"default": '', "multiline": False}),
                "text_d": ("STRING", {"default": '', "multiline": False}),
            }
        }
    TEXT_TYPE = "STRING"
    BOOLEAN_TYPE = "BOOLEAN"
    

    RETURN_TYPES = (TEXT_TYPE,TEXT_TYPE,TEXT_TYPE,TEXT_TYPE,TEXT_TYPE,TEXT_TYPE,)
    FUNCTION = "generate_date_string"

    CATEGORY = "RenderRiftNodes"

    def generate_date_string(self, text='', enable_date_folder=False, text_c='', text_d='', int_1=0):
        
        tokens = TextTokens()
        
        lq_video_path = datetime.now().strftime("%Y-%m-%d") + "/"
        lq_img_path = datetime.now().strftime("%Y-%m-%d") + "/"
        
        hq_video_path = datetime.now().strftime("%Y-%m-%d") + "/"
        hq_img_path = datetime.now().strftime("%Y-%m-%d") + "/"
        
        fd_video_path = datetime.now().strftime("%Y-%m-%d") + "/"
        fd_img_path = datetime.now().strftime("%Y-%m-%d") + "/"
        
        def append_text(base_text, new_text):
            return base_text + new_text

        lq_video_path = append_text(lq_video_path, str(int_1) + "lq_")
        lq_img_path = append_text(lq_img_path, str(int_1) + "_lq/lqimg_")
            
        lq_video_path = tokens.parseTokens(lq_video_path)
        lq_img_path = tokens.parseTokens(lq_img_path)
        
        hq_video_path = append_text(hq_video_path, str(int_1) + "hq_")
        hq_img_path = append_text(hq_img_path, str(int_1) + "_hq/hqimg_")
        
        fd_video_path = append_text(fd_video_path, str(int_1) + "hq_")
        fd_img_path = append_text(fd_img_path, str(int_1) + "_hq/hqimg_")
            
        hq_video_path = tokens.parseTokens(hq_video_path)
        hq_img_path = tokens.parseTokens(hq_img_path)
        
        fd_video_path = tokens.parseTokens(fd_video_path)
        fd_img_path = tokens.parseTokens(fd_img_path)
        
        return (lq_video_path,lq_img_path, hq_video_path, hq_img_path, fd_video_path, fd_img_path,)

class AnalyseMetadata:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            }
        }

    CATEGORY = "RenderRiftNodes"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyse_metadata"

    def analyse_metadata(self, image):
        _, metadata = self.load_image_and_extract_metadata(image)
        node_types = self.extract_node_types(metadata)
        overlayed_image_path = self.overlay_metadata_on_image(image)
        return overlayed_image_path if overlayed_image_path else "Failed to overlay metadata"

    def extract_node_types(self, metadata):
        try:
            metadata_dict = json.loads(metadata)
            node_types = [node['class_type'] for node in metadata_dict.values()]
            return node_types
        except json.JSONDecodeError:
            return ["Error parsing metadata"]

    def load_image_and_extract_metadata(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            metadata = img.info  # Extract metadata
            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img)[None,]  # Convert to tensor
        return img_tensor, metadata.get('Description', '{}')  # Return Description or empty dict

    @classmethod
    def IS_CHANGED(cls, image):
        # Implement as needed
        pass

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class LoadImageWithMeta:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "RenderRiftNodes"

    RETURN_TYPES = ("IMAGE", "MASK","METADATA")
    FUNCTION = "load_image_with_meta"
    def load_image_with_meta(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        
        # Extract metadata (example: EXIF data)
        metadata = i.info
        # print(metadata)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0), metadata)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True