import lmdb
import caffe
import numpy as np
from PIL import Image

class LMDBWrapperBase(object):
    def __init__(self, lmdb_name, init_sample_estimate=1000, sample_size=64):
        self.lmdb_name = lmdb_name
        self.init_sample_estimate = init_sample_estimate
        self.sample_size = sample_size
        self.env = lmdb.open(lmdb_name, map_size=init_sample_estimate * sample_size)
        self.idx = self.env.stat()['entries'] - 1

    def keygen(self, idx):
        raise NotImplementedError(
            "Implement a keygen function to map an index to a unique key. Check LMDBWrapper class for default keygen")
        pass

    def read(self, key=None, idx=None):
        if key is None and idx is None:
            return None
        key = self.keygen(idx) if key is None else key
        with self.env.begin() as txn:
            value = txn.get(key)
        return value

    def write(self, key=None, value=''):
        self.idx += 1 if key is None and idx is None else 0
        key = self.keygen(self.idx) if key is None else key
        # key = self.keygen(idx) if idx is not None else key
        try:
            with self.env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError:
            txn.abort()
            current_map_size = self.env.info()['map_size']
            print 'increasing size of lmdb {} from {} MB to {} MB'.format(
                self.env.path(), current_map_size / 1e6, 2 * current_map_size / 1e6)
            self.env.set_mapsize(current_map_size * 2)
            self.write(key, value)

    def __del__(self):
        self.env.close()


class LMDBWrapper(LMDBWrapperBase):
    def __init__(self, lmdb_name, init_sample_estimate=1000, sample_size=64):
        LMDBWrapperBase.__init__(self, lmdb_name=lmdb_name,
                                 init_sample_estimate=init_sample_estimate, sample_size=sample_size)

    def keygen(self, idx):
        return '{:08}'.format(idx).encode('ascii')


class LMDBWrapperCaffe(LMDBWrapper):
    def __init__(self, lmdb_name, init_sample_estimate=250 * 400, img_size=(224, 224, 3), channels_first=False):
        LMDBWrapper.__init__(
            self, lmdb_name, init_sample_estimate=init_sample_estimate, sample_size=np.prod(img_size))
        self.img_size = (224, 224, 3)
        self.channels_first = channels_first

    def read_img(self, idx):
        value = self.read(idx=idx)
        if value is None:
            return None, None
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        im = np.fromstring(datum.data, dtype='uint8')
        im = im.reshape(datum.channels, datum.height, datum.width)
        if not self.channels_first:
            im = im.transpose(1, 2, 0)
        return im, datum.label

    def write_img(self, img, label, idx=None):
        if not self.channels_first:
            img = img.transpose(2, 0, 1)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = img.shape[0]
        datum.height = img.shape[1]
        datum.width = img.shape[2]
        datum.data = img.tobytes()
        datum.label = label
        key = self.keygen(self.idx) if idx is None else self.keygen(idx)
        self.idx = self.idx + 1 if idx is None else self.idx
        self.write(key=key, value=datum.SerializeToString())


def view(db, view_label=None):
    """
    Pass a LMDBWrapperCaffe object
    """
    if not isinstance(db, LMDBWrapperCaffe):
        print 'view_lmdb works only for LMDBWrapperCaffe objects'
        return None
    print 'viewing', db.lmdb_name
    with db.env.begin() as txn:
        while True:
            idx = np.random.randint(0, db.env.stat()['entries'])
            # print idx
            im, label = db.read_img(idx)
            if im is None:
                continue
            im = im.transpose(1, 2, 0) if db.channels_first else im
            vlabel = label if view_label is None else view_label
            if label == vlabel:
                Image.fromarray(im[:, :, ::-1]).show()
                q = raw_input('label {}. Hit q to quit '.format(label))                             
                if q == 'q' or q == 'Q':
                    break

lmdb_path = "/home/buralako/dataset-txt/ICDAR13/Challenge02/Localization/ICDAR13/lmdb/ICDAR13_trainval_lmdb"
lmdbwrapper = LMDBWrapperCaffe(lmdb_path, img_size=(300, 300, 3), channels_first=True)
view(lmdbwrapper)
