import hls4ml
import yaml
from qkeras import QActivation, QConv2D, QDense
from tensorflow import keras


def main(params):
    keras_model = keras.models.load_model(
        params["cicadaEncoderOnly"]["outputModel"],
        custom_objects={
            "QConv2D": QConv2D,
            "QDense": QDense,
            "QActivation": QActivation,
        },
    )

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")

    hls_config["Model"]["Strategy"] = "Latency"

    # for layer_name in hls_config['LayerName']:
    #     if 'conv' in layer_name:
    #         hls_config['LayerName'][layer_name]['StreamOutputs'] = False
    #         hls_config['LayerName'][layer_name]['implementation'] = 'array'

    # # Default reuse factor for all layers
    for layer in hls_config["LayerName"].keys():
        hls_config["LayerName"][layer]["ReuseFactor"] = 2

    # hls_config["LayerName"]["student_input"]["Precision"]["result"] = "fixed<10,6>"

    # Conv 1 layer
    hls_config["LayerName"]["conv_1"]["StreamOutputs"] = False
    hls_config["LayerName"]["conv_1"]["implementation"] = "array"
    hls_config["LayerName"]["conv_1"]["Precision"]["result"] = "fixed<12,8>"
    hls_config["LayerName"]["conv_1"]["Precision"]["accum"] = "fixed<30,22>"
    hls_config["LayerName"]["conv_1"]["Strategy"] = "Resource"
    hls_config["LayerName"]["conv_1"]["ReuseFactor"] = 1
    hls_config["LayerName"]["conv_1"]["ParallelizationFactor"] = 21

    # conv_2 layer
    hls_config["LayerName"]["conv_2"]["StreamOutputs"] = False
    hls_config["LayerName"]["conv_2"]["implementation"] = "array"
    hls_config["LayerName"]["conv_2"]["Precision"]["result"] = "fixed<30,22>"
    hls_config["LayerName"]["conv_2"]["Precision"]["accum"] = "fixed<30,22>"
    hls_config["LayerName"]["conv_2"]["Strategy"] = "Resource"
    hls_config["LayerName"]["conv_2"]["ReuseFactor"] = 1
    hls_config["LayerName"]["conv_2"]["ParallelizationFactor"] = 21

    # dense outputs
    hls_config["LayerName"]["z_mu"]["Precision"]["result"] = "fixed<16,8>"
    hls_config["LayerName"]["z_mu"]["Precision"]["accum"] = "fixed<26,14>"

    # Okay... how precise do these need to be in our outputs now?

    # hls_config["LayerName"]["conv"]["Strategy"] = "Resource"
    # hls_config["LayerName"]["conv"]["ReuseFactor"] = 1
    # hls_config["LayerName"]["conv"]["ParallelizationFactor"] = 21
    # hls_config["LayerName"]["conv"]["Precision"]["result"] = "fixed<30,22>"
    # hls_config["LayerName"]["conv"]["Precision"]["accum"] = "fixed<30,22>"

    # # Dense1 precision (v2)
    # hls_config["LayerName"]["dense1"]["Precision"]["result"] = "fixed<26,14>"
    # hls_config["LayerName"]["dense1"]["Precision"]["accum"] = "fixed<26,14>"

    # # ---- Dense2 output precision ----
    # hls_config["LayerName"]["dense2"]["Precision"]["result"] = "fixed<26,14>"
    # hls_config["LayerName"]["dense2"]["Precision"]["accum"] = "fixed<26,14>"

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        clock_period=6.25,
        backend="Vitis",
        hls_config=hls_config,
        io_type="io_parallel",
        output_dir="encoder_only_firmware",
        part="xc7vx690tffg1927-2",
        project_name="cicada",
        version=3,
    )
    hls_model.compile()


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)
