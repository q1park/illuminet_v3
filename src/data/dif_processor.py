import os
import re
from src.data.dif_reader import dif_extract
from src.data.structures import LineData, Processor

class DIFProcessor(Processor):
    def __init__(self, data_dir, **kwargs):
        super(DIFProcessor, self).__init__(**kwargs)
        self.dir = os.path.join(*re.split(r'[\/\\]', data_dir))
        self.toc_dir = os.path.join(self.dir, 'Player', 'CourseDirectory')
        self.page_dir = os.path.join(self.dir, 'Player', 'Pages')
        self.__dict__.update(**kwargs)
        
    def load_dif(self):
        for d in dif_extract(self.toc_dir, self.page_dir):
            self.add(**{d['name']:LineData(**d)})