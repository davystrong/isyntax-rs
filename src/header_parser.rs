use base64::prelude::*;
use macro_rules_attribute::derive;
use paste::paste;
use serde::Deserialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum ParseError {
    #[error("empty text: {group}, {element}")]
    EmptyText { group: String, element: String },
    #[error("empty child: {group}, {element}")]
    EmptyChild { group: String, element: String },
    #[error("invalid number: {0}")]
    InvalidNumber(String),
    #[error("invalid number array: {0}")]
    InvalidNumberArray(String),
    #[error("unknown attribute: {group}, {element}")]
    UnknownAttribute { group: String, element: String },
    #[error("unknown data object: {0}")]
    UnknownDataObject(String),
    #[error("unexpeted data object type in this location")]
    UnexpectedDataObjectType,
    #[error("missing attribute \"{attribute}\" for \"{data_object}\"")]
    MissingAttribute {
        attribute: String,
        data_object: String,
    },
    #[error("infallible")]
    Infallible(#[from] std::convert::Infallible),
    #[error("base64 decode error")]
    Base64DecodeError(#[from] base64::DecodeError),
    #[error("hex decode error")]
    HexDecodeError(#[from] std::num::ParseIntError),
    #[error("error loading image into memory")]
    ImageLoadError(#[from] image::ImageError),
    #[error("invalid attribute")]
    InvalidAttribute,
}

type Result<T> = std::result::Result<T, ParseError>;

mod raw_header {
    use serde::Deserialize;

    #[derive(Deserialize, Debug)]
    pub(crate) struct Attribute {
        #[serde(rename = "@Name")]
        pub(crate) name: String,

        #[serde(rename = "@Group")]
        pub(crate) group: String,

        #[serde(rename = "@Element")]
        pub(crate) element: String,

        #[serde(rename = "@PMSVR")]
        pub(crate) pmsvr: String,

        #[serde(rename = "$text")]
        pub(crate) text: Option<String>,

        #[serde(rename = "$value")]
        pub(crate) child: Option<Array>,
    }

    #[derive(Deserialize, Debug)]
    pub(crate) struct DataObject {
        #[serde(rename = "@ObjectType")]
        pub(crate) object_type: String,

        #[serde(rename = "Attribute")]
        pub(crate) attributes: Vec<Attribute>,
        // #[serde(rename = "Array")]
        // arrays: Vec<Array>,

        // #[serde(rename = "DataObject")]
        // data_objects: Vec<DataObject>,
    }

    #[derive(Deserialize, Debug)]
    pub(crate) struct Array {
        #[serde(rename = "DataObject")]
        pub(crate) data_objects: Vec<DataObject>,
        // This doesn't seem to appear
        // #[serde(rename = "Attribute")]
        // attributes: Option<Vec<Attribute>>,
    }
}

mod enum_header {
    use super::raw_header;
    use super::{ParseError, Result};
    use crate::header_parser::enum_header;
    use macro_rules_attribute::derive;
    use serde::Deserialize;
    use std::str::FromStr;
    use paste::paste;

    macro_rules! TryFromRawAttribute {
        ($( #[$($attrs:tt)*] )*
        $pub:vis enum $enum_name:ident {
            $($variant:ident($inner:ty) = $value:literal,)*
        }) => {
            impl TryFrom<raw_header::Attribute> for Attribute {
                type Error = ParseError;

                fn try_from(
                    value: raw_header::Attribute,
                ) -> std::result::Result<Self, Self::Error> {
                    let mut match_val = value.group.to_uppercase().replace("0X", "");
                    match_val.push_str(value.element.to_uppercase().replace("0X", "").as_str());
                    let match_val = u32::from_str_radix(&match_val, 16)?;
                    match match_val {
                        $($value => {
                            Ok(Attribute::$variant(Parse::parse(value)?))
                        },)*
                        _ => Err(ParseError::UnknownAttribute {
                            group: value.group.to_owned(),
                            element: value.element.to_owned(),
                        }),
                    }
                }
            }
        };
    }

    #[derive(Debug, Deserialize, TryFromRawAttribute!)]
    #[serde(try_from = "raw_header::Attribute")]
    #[repr(u32)]
    pub(crate) enum Attribute {
        DicomAcquisitionDatetime(String) = 0x0008002A,
        DicomManufacturer(String) = 0x00080070,
        DicomManufacturersModelName(String) = 0x00081090,
        DicomDerivationDescription(String) = 0x00082111,
        DicomDeviceSerialNumber(String) = 0x00181000,
        DicomSoftwareVersions(Vec<String>) = 0x00181020,
        DicomDateOfLastCalibration(Vec<String>) = 0x00181200,
        DicomTimeOfLastCalibration(Vec<String>) = 0x00181201,
        DicomSamplesPerPixel(u16) = 0x00280002,
        DicomBitsAllocated(u16) = 0x00280100,
        DicomBitsStored(u16) = 0x00280101,
        DicomHighBit(u16) = 0x00280102,
        DicomPixelRepresentation(u16) = 0x00280103,
        DicomIccprofile(String) = 0x00282000,
        DicomLossyImageCompression(String) = 0x00282110,
        DicomLossyImageCompressionRatio(Vec<f64>) = 0x00282112,
        DicomLossyImageCompressionMethod(String) = 0x00282114,
        PiimDpScannerRackNumber(u16) = 0x101D1007,
        PiimDpScannerSlotNumber(u16) = 0x101D1008,
        PiimDpScannerOperatorId(String) = 0x101D1009,
        PiimDpScannerCalibrationStatus(String) = 0x101D100A,
        PimDpUfsInterfaceVersion(String) = 0x301D1001,
        PimDpUfsBarcode(String) = 0x301D1002,
        PimDpScannedImages(Vec<DataObject>) = 0x301D1003,
        PimDpImageType(String) = 0x301D1004,
        PimDpImageData(String) = 0x301D1005,
        PimDpScannerRackPriority(u16) = 0x301D1010,
        DpColorManagement(Vec<DataObject>) = 0x301D1013,
        DpImagePostProcessing(Vec<DataObject>) = 0x301D1014,
        DpSharpnessGainRgb24(f64) = 0x301D1015,
        DpClaheClipLimitY16(f64) = 0x301D1016,
        DpClaheNrBinsY16(u16) = 0x301D1017,
        DpClaheContextDimensionY16(u16) = 0x301D1018,
        DpWaveletQuantizerSettingsPerColor(Vec<DataObject>) = 0x301D1019,
        DpWaveletQuantizerSettingsPerLevel(Vec<DataObject>) = 0x301D101A,
        DpWaveletQuantizer(u16) = 0x301D101B,
        DpWaveletDeadzone(u16) = 0x301D101C,
        UfsImageGeneralHeaders(Vec<DataObject>) = 0x301D2000,
        UfsImageNumberOfBlocks(u32) = 0x301D2001,
        UfsImageDimensionsOverBlock(Vec<u16>) = 0x301D2002,
        UfsImageDimensions(Vec<DataObject>) = 0x301D2003,
        UfsImageDimensionName(String) = 0x301D2004,
        UfsImageDimensionType(String) = 0x301D2005,
        UfsImageDimensionUnit(String) = 0x301D2006,
        UfsImageDimensionScaleFactor(f64) = 0x301D2007,
        UfsImageDimensionDiscreteValuesString(Vec<String>) = 0x301D2008,
        UfsImageBlockHeaderTemplates(Vec<DataObject>) = 0x301D2009,
        UfsImageDimensionRanges(Vec<DataObject>) = 0x301D200A,
        UfsImageDimensionRange(Vec<i32>) = 0x301D200B, // The spec says u32 but in practice it's i32
        UfsImageDimensionsInBlock(Vec<u16>) = 0x301D200C,
        UfsImageBlockHeaders(Vec<DataObject>) = 0x301D200D,
        UfsImageBlockCoordinate(Vec<u32>) = 0x301D200E,
        UfsImageBlockCompressionMethod(String) = 0x301D200F,
        UfsImageBlockDataOffset(u64) = 0x301D2010,
        UfsImageBlockSize(u64) = 0x301D2011,
        UfsImageBlockHeaderTemplateId(u32) = 0x301D2012,
        UfsImagePixelTransformationMethod(String) = 0x301D2013,
        UfsImageBlockHeaderTable(String) = 0x301D2014,
    }

    trait Parse: Sized {
        fn parse(value: raw_header::Attribute) -> Result<Self>;

        fn parse_string(value: raw_header::Attribute) -> Result<String> {
            Ok(value.text.unwrap_or(String::new()))
        }

        fn parse_string_array(value: raw_header::Attribute) -> Result<Vec<String>> {
            let mut value = value.text.ok_or(ParseError::EmptyText {
                group: value.group,
                element: value.element,
            })?;
            value.remove(0);
            value.pop();
            Ok(value.split("\" \"").map(str::to_owned).collect())
        }

        fn parse_data_object_array(value: raw_header::Attribute) -> Result<Vec<DataObject>> {
            let value = value.child.ok_or(ParseError::EmptyChild {
                group: value.group,
                element: value.element,
            })?;
            value
                .data_objects
                .into_iter()
                .map(|d| d.try_into())
                .collect()
        }

        fn parse_number<T: FromStr>(value: raw_header::Attribute) -> Result<T> {
            let value = value.text.ok_or(ParseError::EmptyText {
                group: value.group,
                element: value.element,
            })?;
            value.parse().map_err(|_| ParseError::InvalidNumber(value))
        }

        fn parse_number_array<T: FromStr>(value: raw_header::Attribute) -> Result<Vec<T>> {
            let mut value = value.text.ok_or(ParseError::EmptyText {
                group: value.group,
                element: value.element,
            })?;
            if value.chars().nth(0) == Some('"') && value.chars().last() == Some('"') {
                value.remove(0);
                value.pop();
            }
            let result: Result<Vec<_>> = value
                .split(" ")
                .map(|s| {
                    s.parse()
                        .map_err(|_| ParseError::InvalidNumber(s.to_owned()))
                })
                .collect();
            result.map_err(|_| ParseError::InvalidNumberArray(value))
        }
    }

    macro_rules! impl_parse {
        ($type:ty, $impl:ident) => {
            paste! {
                impl Parse for $type {
                    fn parse(value: raw_header::Attribute) -> Result<Self> {
                        Self::[< parse_ $impl >](value)
                    }
                }
                
                impl Parse for Vec<$type> {
                    fn parse(value: raw_header::Attribute) -> Result<Self> {
                        Self::[< parse_ $impl _array >](value)
                    }
                }
            }
        };
    }

    impl_parse!(String, string);
    impl_parse!(u16, number);
    impl_parse!(u32, number);
    impl_parse!(i32, number);
    impl_parse!(u64, number);
    impl_parse!(f64, number);

    impl Parse for Vec<enum_header::DataObject> {
        fn parse(value: raw_header::Attribute) -> Result<Self> {
            Self::parse_data_object_array(value)
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(try_from = "raw_header::DataObject")]
    pub(crate) enum DataObject {
        DpUfsImport(Vec<Attribute>),
        DpScannedImage(Vec<Attribute>),
        UfsImageDimensionRange(Vec<Attribute>),
        UfsImageBlockHeaderTemplate(Vec<Attribute>),
        UfsImageGeneralHeader(Vec<Attribute>),
        UfsImageDimension(Vec<Attribute>),
        DpColorManagement(Vec<Attribute>),
        DpImagePostProcessing(Vec<Attribute>),
        DpWaveletQuantizerSettingsPerColor(Vec<Attribute>),
        DpWaveletQuantizerSettingsPerLevel(Vec<Attribute>),
        UfsImageBlockHeader(Vec<Attribute>),
    }

    impl TryFrom<raw_header::DataObject> for DataObject {
        type Error = ParseError;

        fn try_from(value: raw_header::DataObject) -> std::result::Result<Self, Self::Error> {
            let attributes: Result<Vec<_>> =
                value.attributes.into_iter().map(|a| a.try_into()).collect();
            let attributes = attributes?;
            match value.object_type.as_str() {
                "DPUfsImport" => Ok(DataObject::DpUfsImport(attributes)),
                "DPScannedImage" => Ok(DataObject::DpScannedImage(attributes)),
                "UFSImageDimensionRange" => Ok(DataObject::UfsImageDimensionRange(attributes)),
                "UFSImageBlockHeaderTemplate" => {
                    Ok(DataObject::UfsImageBlockHeaderTemplate(attributes))
                }
                "UFSImageBlockHeader" => Ok(DataObject::UfsImageBlockHeader(attributes)),
                "UFSImageGeneralHeader" => Ok(DataObject::UfsImageGeneralHeader(attributes)),
                "UFSImageDimension" => Ok(DataObject::UfsImageDimension(attributes)),
                "DPColorManagement" => Ok(DataObject::DpColorManagement(attributes)),
                "DPImagePostProcessing" => Ok(DataObject::DpImagePostProcessing(attributes)),
                "DPWaveletQuantizerSettingsPerColor" => {
                    Ok(DataObject::DpWaveletQuantizerSettingsPerColor(attributes))
                }
                "DPWaveletQuantizerSettingsPerLevel" => {
                    Ok(DataObject::DpWaveletQuantizerSettingsPerLevel(attributes))
                }
                object_type => Err(ParseError::UnknownDataObject(object_type.to_owned())),
            }
        }
    }
}

macro_rules! TryFromDataObject {
    (muncher $(#[$field_attr:meta])*
        $field_pub:vis $field_name:ident: Vec<$field_type:ty>,
        $($tail:tt)*
    ) => {
        $(#[$field_attr])*
        let $field_name = $field_name.into_iter().map(|it| it.try_into()).collect::<std::result::Result<_, _>>()?;
        TryFromDataObject!{muncher $($tail)*}
    };
    (muncher $(#[$field_attr:meta])*
        $field_pub:vis $field_name:ident: $field_type:ty,
        $($tail:tt)*
    ) => {
        $(#[$field_attr])*
        let $field_name = $field_name;
        TryFromDataObject!{muncher $($tail)*}
    };
    (muncher) => {};
    ($( #[$($attrs:tt)*] )*
    $pub:vis struct $struct_name:ident {
        $($tail:tt)*
    }) => {
        TryFromDataObject! {
            [$($($attrs)*)* custom($struct_name)]
            $pub struct $struct_name {
                $($tail)*
            }
        }
    };
    ([custom($rename:ident) $($attrs:tt)*]
        $pub:vis struct $struct_name:ident {
        $($tail:tt)*
    }) => {
        TryFromDataObject! {
            $rename
            $pub struct $struct_name {
                $($tail)*
            }
        }
    };
    ([$attr:tt $($attrs:tt)*]
        $pub:vis struct $struct_name:ident {
        $($tail:tt)*
    }) => {
        TryFromDataObject! {
            [$($attrs)*]
            $pub struct $struct_name {
                $($tail)*
            }
        }
    };
    ($rename:ident
    $pub:vis struct $struct_name:ident {
        $($tail:tt)*
    }) => {
        impl TryFrom<enum_header::DataObject> for $struct_name {
            type Error = ParseError;

            fn try_from(value: enum_header::DataObject) -> std::result::Result<Self, Self::Error> {
                match value {
                    enum_header::DataObject::$rename(attributes) => {
                        TryFromDataObject!{lets_and_match $rename attributes $($tail)*}
                        TryFromDataObject!{muncher $($tail)*}
                        TryFromDataObject!{output $($tail)*}
                    }
                    _ => Err(ParseError::UnexpectedDataObjectType),
                }
            }
        }
    };
    (lets_and_match $struct_name:ident $attrs:ident $($(#[$field_attr:meta])*
        $field_pub:vis $field_name:ident: $field_type:ty),*$(,)?
        ) => {
            $(let mut $field_name = None;)*
            for att in $attrs {
                match att {
                    $(
                        paste! {enum_header::Attribute::[<$field_name:camel>](value)} => {
                            $field_name = Some(value)
                        }
                    )*
                    _ => {
                        //TODO: Fail if unexpected value found
                    }
                }
            }

            $(let $field_name = $field_name.ok_or_else(|| {
                ParseError::MissingAttribute {
                    attribute: stringify!($field_name).to_owned(),
                    data_object: stringify!($struct_name).to_owned(),
                }
            })?;)*
    };
    (output $($(#[$field_attr:meta])*
        $field_pub:vis $field_name:ident: $field_type:ty),*$(,)?
        ) => {
            Ok(Self{
                $($field_name,)*
            })
    };
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct DpUfsImport {
    pub dicom_manufacturer: String,
    pub dicom_acquisition_datetime: String,
    pub dicom_manufacturers_model_name: String,
    pub dicom_device_serial_number: String,
    pub dicom_software_versions: Vec<String>,
    pub dicom_date_of_last_calibration: Vec<String>,
    pub dicom_time_of_last_calibration: Vec<String>,
    pub piim_dp_scanner_rack_number: u16,
    pub piim_dp_scanner_slot_number: u16,
    pub pim_dp_ufs_interface_version: String,
    pub pim_dp_ufs_barcode: String,
    pub pim_dp_scanned_images: Vec<DpScannedImage>,
    pub piim_dp_scanner_operator_id: String,
    pub piim_dp_scanner_calibration_status: String,
    pub pim_dp_scanner_rack_priority: u16,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(DpScannedImage)]
pub struct WsiImage {
    pub dicom_derivation_description: String,
    pub dicom_lossy_image_compression: String,
    pub dicom_lossy_image_compression_ratio: Vec<f64>,
    pub dicom_lossy_image_compression_method: String,
    pub dp_color_management: Vec<DpColorManagement>,
    pub dp_wavelet_quantizer_settings_per_color: Vec<DpWaveletQuantizerSettingsPerColor>,
    pub ufs_image_general_headers: Vec<UfsImageGeneralHeader>,
    pub ufs_image_block_header_templates: Vec<UfsImageBlockHeaderTemplate>,
    // pub ufs_image_block_headers: Vec<UfsImageBlockHeader>,
    pub ufs_image_block_header_table: String,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(DpScannedImage)]
pub struct LabelImage {
    pub pim_dp_image_data: String,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(DpScannedImage)]
pub struct MacroImage {
    pub pim_dp_image_data: String,
}

pub trait Base64Image {
    fn image_data(&self) -> &str;

    fn decode(&self) -> Result<image::RgbImage> {
        let image_bytes =
            BASE64_STANDARD.decode(self.image_data().replace("\n", "").replace("\r", ""))?;

        let img = image::load_from_memory(&image_bytes)?;
        Ok(img.into_rgb8())
    }
}

impl Base64Image for LabelImage {
    fn image_data(&self) -> &str {
        &self.pim_dp_image_data
    }
}

impl Base64Image for MacroImage {
    fn image_data(&self) -> &str {
        &self.pim_dp_image_data
    }
}

#[derive(Debug, Deserialize)]
pub enum DpScannedImage {
    Wsi(WsiImage),
    Label(LabelImage),
    Macro(MacroImage),
}

impl TryFrom<enum_header::DataObject> for DpScannedImage {
    type Error = ParseError;

    fn try_from(value: enum_header::DataObject) -> std::result::Result<Self, Self::Error> {
        match &value {
            enum_header::DataObject::DpScannedImage(attributes) => {
                let image_type = attributes
                    .iter()
                    .find_map(|a| match a {
                        enum_header::Attribute::PimDpImageType(image_type) => Some(image_type),
                        _ => None,
                    })
                    .ok_or_else(|| ParseError::MissingAttribute {
                        attribute: "pim_dp_image_type".to_owned(),
                        data_object: "DpScannedImage".to_owned(),
                    })?;
                match image_type.as_str() {
                    "LABELIMAGE" => Ok(Self::Label(value.try_into()?)),
                    "MACROIMAGE" => Ok(Self::Macro(value.try_into()?)),
                    "WSI" => Ok(Self::Wsi(value.try_into()?)),
                    _ => Err(ParseError::InvalidAttribute),
                }
            }
            _ => Err(ParseError::UnexpectedDataObjectType),
        }
    }
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct UfsImageGeneralHeader {
    pub ufs_image_number_of_blocks: u32,
    pub ufs_image_dimensions_over_block: Vec<u16>,
    pub ufs_image_dimensions: Vec<UfsImageDimension>,
    pub ufs_image_dimension_ranges: Vec<UfsImageDimensionRange>,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(UfsImageDimension)]
pub struct SpatialDimension {
    pub ufs_image_dimension_name: String,
    pub ufs_image_dimension_unit: String,
    pub ufs_image_dimension_scale_factor: f64,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(UfsImageDimension)]
pub struct ColourComponentDimension {
    pub ufs_image_dimension_name: String,
    pub ufs_image_dimension_discrete_values_string: Vec<String>,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(UfsImageDimension)]
pub struct ScaleDimension {
    // Not sure what the point of this is, but it's literally empty
    pub ufs_image_dimension_name: String,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[custom(UfsImageDimension)]
pub struct WaveletCoefficientDimension {
    pub ufs_image_dimension_name: String,
    pub ufs_image_dimension_discrete_values_string: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub enum UfsImageDimension {
    Spatial(SpatialDimension),
    ColourComponent(ColourComponentDimension),
    Scale(ScaleDimension),
    WaveletCoefficient(WaveletCoefficientDimension),
}

impl TryFrom<enum_header::DataObject> for UfsImageDimension {
    type Error = ParseError;

    fn try_from(value: enum_header::DataObject) -> std::result::Result<Self, Self::Error> {
        match &value {
            enum_header::DataObject::UfsImageDimension(attributes) => {
                let dimension_type = attributes
                    .iter()
                    .find_map(|a| match a {
                        enum_header::Attribute::UfsImageDimensionType(dimension_type) => {
                            Some(dimension_type)
                        }
                        _ => None,
                    })
                    .ok_or_else(|| ParseError::MissingAttribute {
                        attribute: "ufs_image_dimension_type".to_owned(),
                        data_object: "UfsImageDimension".to_owned(),
                    })?;
                match dimension_type.as_str() {
                    "spatial" => Ok(Self::Spatial(value.try_into()?)),
                    "colour component" => Ok(Self::ColourComponent(value.try_into()?)),
                    "scale" => Ok(Self::Scale(value.try_into()?)),
                    "waveletcoef" => Ok(Self::WaveletCoefficient(value.try_into()?)),
                    _ => Err(ParseError::InvalidAttribute),
                }
            }
            _ => Err(ParseError::UnexpectedDataObjectType),
        }
    }
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct UfsImageDimensionRange {
    pub ufs_image_dimension_range: Vec<i32>,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct UfsImageBlockHeaderTemplate {
    pub dicom_samples_per_pixel: u16,
    pub dicom_bits_allocated: u16,
    pub dicom_bits_stored: u16,
    pub dicom_high_bit: u16,
    pub dicom_pixel_representation: u16,
    pub ufs_image_dimension_ranges: Vec<UfsImageDimensionRange>,
    pub ufs_image_dimensions_in_block: Vec<u16>,
    pub ufs_image_block_compression_method: String,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct UfsImageBlockHeader {
    pub ufs_image_block_coordinate: Vec<u32>,
    pub ufs_image_block_header_template_id: u32,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct DpColorManagement {
    pub dicom_iccprofile: String,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct DpWaveletQuantizerSettingsPerColor {
    pub dp_wavelet_quantizer_settings_per_level: Vec<DpWaveletQuantizerSettingsPerLevel>,
}

#[derive(Debug, Deserialize, TryFromDataObject!)]
#[serde(try_from = "enum_header::DataObject")]
pub struct DpWaveletQuantizerSettingsPerLevel {
    pub dp_wavelet_quantizer: u16,
    pub dp_wavelet_deadzone: u16,
}

macro_rules! select_image {
    ($base:ident $selector:ident $output:ident) => {
        paste! {
            select_image!([< $base >] iter $selector self [&self] &$output);
            select_image!([< $base _mut >] iter_mut $selector self [&mut self] &mut $output);
            select_image!([< into_ $base >] into_iter $selector self [self] $output);
        }
    };
    ($base:ident $iter:ident $selector:ident $self:ident [$($input:tt)+] $output:ty) => {
        pub fn $base($($input)*) -> $output {
            $self.pim_dp_scanned_images
                .$iter()
                .find_map(|img| {
                    if let DpScannedImage::$selector(inner) = img {
                        Some(inner)
                    } else {
                        None
                    }
                })
                .unwrap()
        }
    };
}

impl DpUfsImport {
    select_image!(wsi Wsi WsiImage);
    select_image!(label_image Label LabelImage);
    select_image!(macro_image Macro MacroImage);
}
