# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ArraySpec(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArraySpec()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsArraySpec(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ArraySpec
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArraySpec
    def Dims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Dimension import Dimension
            obj = Dimension()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ArraySpec
    def DimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ArraySpec
    def DimsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def ArraySpecStart(builder):
    builder.StartObject(1)

def Start(builder):
    ArraySpecStart(builder)

def ArraySpecAddDims(builder, dims):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(dims), 0)

def AddDims(builder, dims):
    ArraySpecAddDims(builder, dims)

def ArraySpecStartDimsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartDimsVector(builder, numElems: int) -> int:
    return ArraySpecStartDimsVector(builder, numElems)

def ArraySpecEnd(builder):
    return builder.EndObject()

def End(builder):
    return ArraySpecEnd(builder)
