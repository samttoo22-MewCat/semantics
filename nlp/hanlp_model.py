# HanLP Model definition
# - will automatically download pretrained model weights
from hanlp import hanlp

hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_UDEP_SDP_CON_ELECTRA_SMALL_ZH)
