"""A plain ComfyUI multiline STRING source with an optional browser-side find bar."""


class SearchableMultilineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output_text"
    CATEGORY = "Amin/Text"

    def output_text(self, text):
        return (text,)


NODE_CLASS_MAPPINGS = {"SearchableMultilineText": SearchableMultilineText}
NODE_DISPLAY_NAME_MAPPINGS = {"SearchableMultilineText": "Text (Multiline + Search)"}
