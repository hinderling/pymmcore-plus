from __future__ import annotations

from abc import abstractmethod
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from pymmcore_plus.core._constants import DeviceType, Keyword, PixelFormat

from ._device_base import Device

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from numpy.typing import DTypeLike


class CameraDevice(Device):
    # mandatory methods for Camera device adapters

    _TYPE: ClassVar[Literal[DeviceType.Camera]] = DeviceType.Camera

    @abstractmethod
    def get_exposure(self) -> float:
        """Get the current exposure time in milliseconds."""
        ...

    @abstractmethod
    def set_exposure(self, exposure: float) -> None:
        """Set the exposure time in milliseconds."""
        ...

    @abstractmethod
    def sensor_shape(self) -> tuple[int, ...]:
        """Return the full sensor shape of the image buffer.

        This is used when querying the full sensor dimensions.
        If the camera is grayscale, it should return (height, width).
        If the camera is color, it should return (height, width, n_channels).
        """

    @abstractmethod
    def dtype(self) -> DTypeLike:
        """Return the data type of the image buffer."""

    def snap(self, buffer: np.ndarray) -> Mapping:
        """Snap a single image into the provided buffer.

        Fill the provided buffer with image data and return metadata.
        The buffer will be sized to ``sensor_shape()`` (full frame).

        Parameters
        ----------
        buffer : np.ndarray
            Pre-allocated buffer to fill with image data. Shaped to
            ``sensor_shape()``.

        Returns
        -------
        Mapping
            Metadata for the acquired image.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement snap() or start_sequence()"
        )

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:
        """Start a sequence acquisition.

        The default implementation calls ``snap()`` for each frame. If the
        camera has an active ROI, the default will snap into a reusable
        full-frame buffer and copy the cropped region into the output buffer.

        Override this method only if your camera needs full control over
        the acquisition loop (e.g. hardware-triggered sequences).

        Parameters
        ----------
        n : int | None
            If an integer, this is the number of images to acquire.
            If None, the camera should acquire images indefinitely until stopped.
        get_buffer : Callable[[Sequence[int], DTypeLike], np.ndarray]
            A callable that returns a buffer for the camera to fill with image
            data.

        Yields
        ------
        Mapping
            Metadata for each acquired image.
        """
        sensor = self.sensor_shape()
        roi = self._roi
        dtype = self.dtype()

        if roi is None:
            # No ROI — snap directly into the output buffer (zero overhead)
            count = 0
            limit = n if n is not None else 2**63
            while count < limit:
                buf = get_buffer(sensor, dtype)
                meta = self.snap(buf)
                yield meta
                count += 1
        else:
            # ROI active — snap into full-frame buffer, crop into output
            x, y, w, h = roi
            full_buf = np.empty(sensor, dtype=dtype)
            count = 0
            limit = n if n is not None else 2**63
            while count < limit:
                roi_shape = self.shape()
                out = get_buffer(roi_shape, dtype)
                meta = self.snap(full_buf)
                out[:] = full_buf[y : y + h, x : x + w]
                yield meta
                count += 1

    # Standard Properties --------------------------------------------

    # these are the standard properties that cameras may implement.
    # Cameras are not required to implement all of these properties, and they may
    # implement additional properties as well.
    # To implement a property, you MUST define a `get_<snake_name>` method and
    # MAY define a `set_<snake_name>` method.
    # To modify the standard properties, you can use the following methods in your
    # __init__, (after calling super().__init__()):
    # self.set_property_value(name, ...)
    # self.set_property_allowed_values(name, ...)
    # self.set_property_limits(name, ...)
    # self.set_property_sequence_max_length(name, ...)

    STANDARD_PROPERTIES: ClassVar = MappingProxyType(
        {
            Keyword.ActualInterval_ms: ("actual_interval_ms", float),
            Keyword.Binning: ("binning", int),
            Keyword.CameraID: ("camera_id", str),
            Keyword.CameraName: ("camera_name", str),
            Keyword.CCDTemperature: ("ccd_temperature", float),
            Keyword.CCDTemperatureSetPoint: ("ccd_temperature_set_point", float),
            Keyword.EMGain: ("em_gain", float),
            Keyword.Exposure: ("exposure", float),
            Keyword.Gain: ("gain", float),
            Keyword.Interval_ms: ("interval_ms", float),
            Keyword.Offset: ("offset", float),
            # Keyword.PixelType: ("pixel_type", str),  # don't use.  use PixelFormat
            "PixelFormat": ("pixel_format", PixelFormat),
            Keyword.ReadoutMode: ("readout_mode", str),
            Keyword.ReadoutTime: ("readout_time", float),
            Keyword.Metadata_ROI_X: ("roi_x", int),
            Keyword.Metadata_ROI_Y: ("roi_y", int),
        }
    )

    # optional methods
    # def get_camera_name(self) -> str:
    # def set_camera_name(self, value: str) -> None:
    # def get_camera_id(self) -> str:
    # def set_camera_id(self, value: str) -> None:
    # def get_binning(self) -> str:
    # def set_binning(self, value: str) -> None:
    # def get_pixel_format(self) -> PixelFormat: ...
    # def set_pixel_format(self, value: PixelFormat) -> None: ...
    # def get_gain(self) -> str:
    # def set_gain(self, value: str) -> None:
    # def get_offset(self) -> str:
    # def set_offset(self, value: str) -> None:
    # def get_readout_mode(self) -> str:
    # def set_readout_mode(self, value: str) -> None:
    # def get_readout_time(self) -> str:
    # def set_readout_time(self, value: str) -> None:
    # def get_actual_interval_ms(self) -> str:
    # def set_actual_interval_ms(self, value: str) -> None:
    # def get_interval_ms(self) -> str:
    # def set_interval_ms(self, value: str) -> None:
    # def get_em_gain(self) -> str:
    # def set_em_gain(self, value: str) -> None:
    # def get_ccd_temperature(self) -> str:
    # def set_ccd_temperature(self, value: str) -> None:
    # def get_ccd_temperature_set_point(self) -> str:
    # def set_ccd_temperature_set_point(self, value: str) -> None:

    def __init__(self) -> None:
        # Validate that subclass implements snap() or start_sequence()
        if (
            type(self).snap is CameraDevice.snap
            and type(self).start_sequence is CameraDevice.start_sequence
        ):
            raise TypeError(
                f"{type(self).__name__} must implement snap() or start_sequence()"
            )
        super().__init__()
        self.register_standard_properties()

    def register_standard_properties(self) -> None:
        """Inspect the class for standard properties and register them."""
        cls = type(self)
        for name, (snake_name, prop_type) in self.STANDARD_PROPERTIES.items():
            if getter := getattr(cls, f"get_{snake_name}", None):
                setter = getattr(cls, f"set_{snake_name}", None)
                seq_loader = getattr(cls, f"load_{snake_name}_sequence", None)
                seq_starter = getattr(cls, f"start_{snake_name}_sequence", None)
                seq_stopper = getattr(cls, f"stop_{snake_name}_sequence", None)
                self.register_property(
                    name=name,
                    property_type=prop_type,
                    getter=getter,
                    setter=setter,
                    sequence_loader=seq_loader,
                    sequence_starter=seq_starter,
                    sequence_stopper=seq_stopper,
                )

    # ROI support -----------------------------------------------------

    _roi: tuple[int, int, int, int] | None = None

    def shape(self) -> tuple[int, ...]:
        """Return the current image shape, accounting for any active ROI.

        If an ROI is set, returns ``(height, width)`` of the ROI.
        Otherwise returns ``sensor_shape()``.
        """
        if self._roi is not None:
            _, _, w, h = self._roi
            return (h, w)
        return self.sensor_shape()

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return the current ROI as ``(x, y, width, height)``.

        Returns the stored ROI or the full sensor frame if none is set.
        """
        if self._roi is not None:
            return self._roi
        h, w, *_ = self.sensor_shape()
        return (0, 0, w, h)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set the ROI, validating bounds against the sensor shape."""
        h, w, *_ = self.sensor_shape()
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid ROI ({x}, {y}, {width}, {height}): "
                "coordinates must be non-negative and dimensions positive."
            )
        if x + width > w or y + height > h:
            raise ValueError(
                f"ROI ({x}, {y}, {width}, {height}) exceeds sensor bounds ({w}x{h})."
            )
        self._roi = (x, y, width, height)

    def clear_roi(self) -> None:
        """Reset the ROI to the full sensor frame."""
        self._roi = None

    # Standard Properties, default implementations -------------------

    # We always implement a standard binning getter.  It does not
    # mean that the camera supports binning, unless they implement a setter.
    def get_binning(self) -> int:
        """Get the binning factor for the camera."""
        return 1  # pragma: no cover
