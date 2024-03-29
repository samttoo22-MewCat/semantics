# NLP Utilities
# - data structure conversions
class NlpUtils:

  # Convert AMR graph to Dictionary (JSON obj)
  def amr_to_dict(self, graph):
    nodes = []
    edges = []

    for triple in graph.triples:
      if triple[1] == ':instance':
        nodes.append({
          'id': triple[0],
          'concept': triple[2]
        })
      else:
        edges.append({
          'from': triple[0],
          'to': triple[2],
          'relation': triple[1],
        })
    
    return {
      'nodes': nodes,
      'edges': edges,
    }

nlp_utils = NlpUtils()
