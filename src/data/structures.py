import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

@dataclass
class Data:
    """
    Specify the features and types input to the first model layer
    """
    data: Optional[List[Union[tuple, int, float, str]]] = None
    index_map: Optional[dict] = None
        
    def __call__(self, name):
        if self.index_map is None:
            raise KeyError('no index map')
        else:
            return self.data[self.index_map[name]]
        
@dataclass
class MultiData(Data):
    """
    Specify the features and types input to the first model layer
    """
    mode: Optional[str] = None
    vector: Optional[str] = None
        
    token_map: Optional[dict] = None
    size: Optional[int] = None
    min: Optional[Union[int, float, str]] = None
    max: Optional[Union[int, float, str]] = None
    max_len: Optional[int] = None
        
class Processor:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        
    def add(self, **kwargs):
        self.__dict__.update(**kwargs)
        
    def save_pickle(self, pkl):
        with open(os.path.join(self.dir, pkl), "wb") as f:
            pickle.dump(self.__dict__, f)
            
    def load_pickle(self, pkl):
        with open(os.path.join(self.dir, pkl), "rb") as f:
            self.__dict__.update(pickle.load(f))
            
def make_segments(lines):
    segments = {}

    def add_segment(feature, idx):
        if feature is not None:
            if feature not in segments:
                segments[feature] = [idx]
            else:
                segments[feature].append(idx)

    start_row, start_col, feature_idx = 0, 0, 0
    nlines, last = len(lines), None

    for i, line in enumerate(lines):
        if type(line)==tuple:
            for j, span in enumerate(line):
                if span!=last:
                    add_segment(last, ((start_row, start_col), (i,j))) 
                    start_row, start_col = i, j
                last = span
        else:
            if line!=last:
                add_segment(last, (start_row, i))  
                start_row = i
            last = line

    if type(last)==tuple:
        add_segment(last, ((start_row, start_col), (nlines-1, len(lines[-1])))) 
    else:
        add_segment(last, (start_row, nlines))
    return segments

@dataclass
class LineData(Data):
    """
    Specify the features and types input to the first model layer
    """
    name: Optional[str] = None
    lines: Optional[List[Union[tuple, int, float, str]]] = None
    segments: Optional[dict] = None
    segment: Optional[bool] = False
                    
    def __post_init__(self, **kwargs):
        if self.segment:
            self.segments = make_segments(self.lines)
            
    def get_lines(self, start_line, end_line):
        return self.lines[start_line:end_line]
            
    def get_segment(self, seg):
        if isinstance(self.segments[seg][0][0], tuple):
            return [line for s in self.segments[seg] for line in self.get_lines(s[0][0], s[1][0])]
        else:
            return [line for s in self.segments[seg] for line in self.get_lines(*s)]