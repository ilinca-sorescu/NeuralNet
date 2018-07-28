#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <vector>

struct Neuron {
    std::vector<double> weights_;
    double input_;
    double output_;
    double delta_;

    Neuron(int inputs) {
        // Creates a neuron with inputs+1 weights in [-1, 1]. The first weight 
        // corresponds to the bias unit.
        for(int i=0; i <= inputs; ++i)
            weights_.push_back((double)std::rand() / (RAND_MAX/2) - 1);
    }
};

struct Layer {
    std::vector<Neuron> layer_;
       
    Layer(int nneurons, int inputs) {
        // Creates a layer with nneurons neurons. Each neuron has the given
        // number of inputs.
        for(int i=0; i < nneurons; ++i)
            layer_.push_back(Neuron(inputs));
    }
 
    static double sigmoid(double x) {
        return 1.0/(1.0 + exp(-x));
    }

    static double sigmoidDerivative(double x) {
        double aux = exp(-x);
        return aux / ((1 + aux) * (1 + aux));
    }

    std::vector<double> forwardProp(std::vector<double> in) {
        // Returns the list of outputs for this layer (including the bias unit).
        // The input already contains the bias unit.
        std::vector<double> out = {1.0};
        for(auto &neuron : layer_) {
            if(in.size() != neuron.weights_.size())
               throw "Number of inputs to the layer must match the number of weights";
            double a = 0;
            for(int i=0; i != in.size(); ++i)
                a += in[i] * neuron.weights_[i];
            neuron.input_ = a;
            neuron.output_ = sigmoid(a);
            out.push_back(neuron.output_);
        }
        return out;
    }

};

struct TrainingEx {
    std::vector<double> in_;
    double out_;

    TrainingEx(std::vector<double> in, double out) {
        in_ = in;
        out_ = out;
    }
};

class NeuralNet {
    private:
        int ninputs_;
        std::vector<Layer> nn_;
        double learning_rate_;

        void forwardProp(std::vector<double> in) {
            // The input does not contain the bias unit. Adding the bias unit.
            std::vector<double> inputs = {1.0};
            inputs.insert(inputs.end(), in.begin(), in.end());
            for(auto &layer : nn_) {
                inputs = layer.forwardProp(inputs);
            }
        }

        void backProp(double actual, double expected) {
            auto output_neuron = nn_[nn_.size()-1].layer_[0];
            output_neuron.delta_ = (expected - actual) * 
                Layer::sigmoidDerivative(output_neuron.input_);
            for(auto &w : output_neuron.weights_)
                w += learning_rate_ * output_neuron.delta_ *
                    output_neuron.output_;

            std::vector<Layer>::iterator layer = nn_.end() - 2;
            for(; layer != nn_.begin(); --layer)
                for(auto &j : layer->layer_) {
                    double err = 0;
                    double index_k = 0;
                    for(auto &k : (layer+1)->layer_) {
                        err += j.weights_[index_k] * k.delta_;
                        ++index_k;
                    }

                    j.delta_ = Layer::sigmoidDerivative(j.input_) * err;
                    double index_i = 0;
                    for(auto &i : (layer+1)->layer_) {
                        j.weights_[index_i] += learning_rate_ * j.output_ *
                            i.delta_;
                        ++index_i;
                    }
                }
        }

    public:
        NeuralNet(int ni, std::vector<int> architecture, double lr) {
            ninputs_ = ni;
            nn_.push_back(Layer(architecture[0], ni));
            learning_rate_ = lr;
            for(int i=1; i < architecture.size(); ++i)
                nn_.push_back(Layer(architecture[i], architecture[i-1]));
            nn_.push_back(Layer(1, architecture[architecture.size()-1]));
        }

        double getOutput(std::vector<double> in) {
            forwardProp(in);
            return nn_[nn_.size()-1].layer_[0].output_;
        }

        void train(std::vector<TrainingEx> data) {
            for(const auto &d : data) {
                double result = getOutput(d.in_);
                backProp(result, d.out_);
            }
        }

        double validate(std::vector<TrainingEx> data) {
            // Returns sum of squared error for the model over the validation
            // data.
            double sum = 0;
            for(const auto &d : data) {
                double result = getOutput(d.in_);
                printf("%lf %lf\n", result, d.out_);
                sum += (result - d.out_) * (result - d.out_);
            }
            return sum / data.size();
        }
};

std::vector<TrainingEx> getTrainingData(int ninputs, const char* filename) {
    std::vector<TrainingEx> result;
    std::ifstream file(filename);
    while(true) {
        std::vector<double> in;
        for(int i=0; i != ninputs; ++i) {
            double x;
            file >> x;
            in.push_back(x);
        }
        double y;
        file >> y;
        result.push_back(TrainingEx(in, y));
        if(file.eof())
            break;
    }
    return result;
}

int main() {
    // <3
    int ninputs = 2;
    auto hidden_units = {31, 31, 31, 31};
    double learning_rate = 3;
    auto nn = NeuralNet(ninputs, hidden_units, learning_rate);
    nn.train(getTrainingData(ninputs, "cylinder.train"));
    auto valid = nn.validate(getTrainingData(ninputs, "cylinder.validate"));
    printf("validation: %lf\n", valid);
    printf("Predicted volume of cylinder:%lf\n", nn.getOutput({1.0, 0.3}));
    return 0;
}
