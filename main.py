def alignment(source_sequence, tokenized_sequence: List[str], index_type: str = 'str', align_type: str = 'one2many') -> Dict:
    """[summary]
    # this is a function that to align sequcences  that before tokenized and after.

    Parameters
    ----------
    source_sequence : [type]
        this is the original sequence, whose type either can be str or list
    tokenized_sequence : List[str]
        this is the tokenized sequcen, which is a list of tokens.
    index_type : str, optional, default: str
        this indicate whether source_sequence is str or list, by default 'str'
    align_type : str, optional, default: one2many
        there may be several kinds of tokenizer style, 
        one2many: one word in source sequence can be split into multiple tokens 
        many2one: many word in source sequence will be merged into one token
        many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.
    
    useage:
    source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
    tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
    char2token = alignment(source_sequence, tokenized_sequence)
    print(char2token)
    for c, t in char2token.items():
        print(source_sequence[c], tokenized_sequence[t])
    """
    char2token = {}
    if isinstance(source_sequence, str) and align_type == 'one2many':
        source_sequence = source_sequence.lower()
        i, j = 0, 0
        while i < len(source_sequence) and j < len(tokenized_sequence):
            cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
            if source_sequence[i] == ' ':
                i += 1
            elif source_sequence[i: i + length] == cur_token:
                for k in range(length):
                    char2token[i + k] = j
                i += length
                j += 1 
            else:
                assert tokenized_sequence[j].startswith('#')
                length = len(tokenized_sequence[j].lstrip('#'))
                assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                for k in range(length):
                    char2token[i + k] = j
                i += length
                j += 1 
    return char2token

