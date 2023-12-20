from .date_integer_node import DateIntegerNode, MetadataOverlayNode, AnalyseMetadata, LoadImageWithMeta

NODE_CLASS_MAPPINGS = {
    "DateIntegerNode": DateIntegerNode,
    "AnalyseMetadata": AnalyseMetadata,
    "MetadataOverlayNode": MetadataOverlayNode,
    "LoadImageWithMeta": LoadImageWithMeta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DateIntegerNode": "Customer Date Folder Format",
    "MetadataOverlayNode": "Image & Metadata Overlay Node",
    "LoadImageWithMeta": "LoadImageWithMeta",
    
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
