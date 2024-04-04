import json
import penman

from hanlp import hanlp
from hanlp_common import document
from nlp.utils import nlp_utils

hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_UDEP_SDP_CON_ELECTRA_SMALL_ZH)
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
amr_parser_en = hanlp.load(hanlp.pretrained.amr.AMR3_SEQ2SEQ_BART_LARGE)
amr_parser_zh = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ENG_ZHO_XLM_BASE)
class NLPService:
  # Constituency parsing
  async def con(self, text: str) -> str:
    print("- Parsing constituency...")
    return hanlp_model.tasks.con(text)
  
  # Dependency parsing
  async def dep(self, text: str) -> str:
    print("- Parsing word dependencies...")
    return hanlp_model.tasks.dep(text)

  # Semantic Dependency Parsing
  async def sdp(self, text: str) -> str:
    print("- Parsing semantic dependencies...")
    return hanlp_model.tasks.sdp(text)
  
  # Universal Dependency parsing
  async def ud(self, text: str) -> str:
    print("- Parsing ud...")
    return hanlp_model.tasks.ud(text)
  
  # Lemmatization
  async def lem(self, text: str) -> str:
    print("- Parsing lem...")
    return hanlp_model.tasks.lem(text)
  
  # Part-of-Speech tagging
  async def pos(self, text: str) -> str:
    print("- Parsing pos...")
    return hanlp_model.tasks.pos(text)
  
  # Named Entity Recognition
  async def parse_ner(self, text: str) -> str:
    print("- Parsing NER...")
    return hanlp_model.tasks.ner.tag_ner(text, tasks='ner/msra').pretty_print()

  # Tokenise (HanLP model)
  async def tokenise(self, text: str) -> document.Document:
    print("- Tokenise...")
    result = tok([text])
    return result
  
  # Abstract Meaning Representation (English)
  async def parse_amr_en(self, text: str) -> str:
    print("- Parsing AMR...")

    # Plain Text -> AMR String
    amr_string = amr_parser(text)
    print("-- Parsed AMR:")
    print(amr_string)
    

    graph = penman.decode(str(amr_string))
    print("-- Parse AMR OK. Converting to graph.")
    print(graph)

    graph_dict = nlp_utils.amr_to_dict(graph)
    graph_json = json.dumps(graph_dict)

    print(type(graph_json))
    return graph_json

  # Abstract Meaning Representation (Chinese)
  async def parse_amr_zh(self, text: str):
    # Plain Text -> Tokenise
    print("- Tokenising first...")
    tok_result = tok([text])

    # Tokenised -> AMR String
    print("- Parsing AMR...")
    amr_string = amr_parser_zh(tok_result, output_amr=False)
    print("-- Parsed AMR:")
    print(amr_string)
    
    return amr_string
