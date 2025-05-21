from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

def parse_pdf_to_json(file_path: str, page_range: str = None, use_llm: bool = False):
    try:
        config_parser = ConfigParser({
            "output_format": "json",
            "page_range": page_range,
            "use_llm": use_llm,
            "disable_image_extraction": True,
            }
        )
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        result = converter(file_path)
        
        return result
    except Exception as e:
        raise Exception(f"Error parsing PDF file: {str(e)}")
