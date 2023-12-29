from .date_integer_node import DateIntegerNode, MetadataOverlayNode, AnalyseMetadata, LoadImageWithMeta, VideoPathMetaExtraction, DisplayMetaOptions

NODE_CLASS_MAPPINGS = {
    "DateIntegerNode": DateIntegerNode,
    "AnalyseMetadata": AnalyseMetadata,
    "MetadataOverlayNode": MetadataOverlayNode,
    "LoadImageWithMeta": LoadImageWithMeta,
    "VideoPathMetaExtraction": VideoPathMetaExtraction,
    "DisplayMetaOptions": DisplayMetaOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DateIntegerNode": "RR_Date_Folder_Format",
    "MetadataOverlayNode": "RR_Image_Metadata_Overlay",
    "LoadImageWithMeta": "RR_LoadImageWithMeta (Depreciated)",
    "VideoPathMetaExtraction": "RR_VideoPathMetaExtraction",
    "DisplayMetaOptions": "RR_DisplayMetaOptions",
    
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
