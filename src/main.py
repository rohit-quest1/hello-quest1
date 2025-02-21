import asyncio
import sys
from typing import (Any, ClassVar, Dict, Final, List, Mapping, Optional,
                    Sequence, Tuple)

from typing_extensions import Self
from viam.components.camera import *
from viam.media.video import NamedImage, ViamImage
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.proto.component.camera import GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType
from viam.utils import struct_to_dict
from PIL import Image
import numpy as np
import cv2
from matplotlib import cm

class HelloCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("rohit", "hello-quest1"), "hello-camera")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Camera component.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        fields = config.attributes.fields
        if not "image_path" in fields:
            raise Exception("Missing image_path attribute.")
        elif not fields["image_path"].HasField("string_value"):
            raise Exception("image_path must be a string.")
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        attrs = struct_to_dict(config.attributes)
        self.image_path = str(attrs.get("image_path"))
        return super().reconfigure(config, dependencies)

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ViamImage:
        # Open the image and convert it to grayscale
        img = Image.open(self.image_path).convert("L")
        img_np = np.array(img)

        # Normalize the image to range 0-255
        normalized_img = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)

        # Apply heatmap using the 'jet' color map
        heatmap = cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)

        # Convert the heatmap to PIL image
        heatmap_img = Image.fromarray(heatmap)

        # Return the heatmap as a Viam image
        return pil_to_viam_image(heatmap_img, CameraMimeType.JPEG)

    async def get_images(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError()

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Tuple[bytes, str]:
        raise NotImplementedError()

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        raise NotImplementedError()


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

