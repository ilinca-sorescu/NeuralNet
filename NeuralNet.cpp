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
        // Creates a neuron with inputs+1 weights in [0, 1]. The first weight 
        // corresponds to the bias unit.
        for(int i=0; i <= inputs; ++i)
            weights_.push_back((double)std::rand() / RAND_MAX);
        input_ = 0;
        output_ = 0;
        delta_ = 0;
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
        if(std::isinf(aux))
            return 0.0;
        return aux / ((1.0 + aux) * (1.0 + aux));
    }

    static double relu(double x) {
        return std::max(0.01, x);
    }

    static double reluDerivative(double x) {
        if(x > 0.0)
            return 1.0;
        return 0.0;
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
            neuron.output_ = relu(a);
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

        void printDelta() {
            // Debug function
            int index_l = 0;
            for(const auto& l : nn_) {
                printf("layer: %d\n", index_l);
                int index_n = 0;
                for(const auto& n : l.layer_) {
                    printf("neuron: %d %lf\n", index_n, n.delta_);
                    ++index_n;
                }
                ++index_l;
            }
        }

        void printWeights() {
            // Debug function
            int index_l = 0;
            for(const auto& l : nn_) {
                printf("layer: %d\n", index_l);
                int index_n = 0;
                for(const auto& n : l.layer_) {
                    printf("neuron: %d\n", index_n);
                    for(const auto& w : n.weights_)
                        printf("%lf\n", w);
                    ++index_n;
                }
                ++index_l;
            }
        }

        void backProp(double actual, double expected, std::vector<double> in) {
            printWeights();

            auto output_neuron = &nn_[nn_.size()-1].layer_[0];
            output_neuron->delta_ = //(expected - actual) * 
                (actual-expected) *
                Layer::reluDerivative(output_neuron->input_);
          
            //printf("%lf\n", output_neuron->delta_);
            std::vector<Layer>::iterator layer = nn_.end() - 2;
            for(; layer >= nn_.begin(); --layer) {
                int index_j = 1; // 0 is for the bias unit
                for(auto &j : layer->layer_) {
                    double err = 0;
                    for(auto &k : (layer+1)->layer_) {
                        err += k.weights_[index_j] * k.delta_;
                        //printf("%lf\n", k.delta_);
                    }

                    j.delta_ = Layer::reluDerivative(j.input_) * err;
                    //printf("!%lf %lf %lf\n", err, j.input_, 
                    //        Layer::reluDerivative(j.input_));
                    for(auto &i : (layer+1)->layer_) {
                        i.weights_[index_j] += learning_rate_ * j.output_ *
                            i.delta_;
                        //printf("%lf %lf %lf\n", learning_rate_, j.output_,
                        //        i.delta_);
                    }
                    ++index_j;
                }
            }

            // first layer
            for(auto &i : nn_[0].layer_) {
                for(int index=0; index != in.size(); ++index) {
                    i.weights_[index+1] += learning_rate_ * i.delta_ *
                        in[index]; 
                }
            }

            // bias unit
            for(auto &l : nn_) 
                for(auto &i : l.layer_)
                    i.weights_[0] += learning_rate_ * i.delta_;

          printWeights();
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
                backProp(result, d.out_, d.in_);
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

std::vector<TrainingEx> getTrainingData(int ninputs, const char* filename,
        double scaling_factors[]) {
    // scaling_factors contains ninputs+1 items, each for scaling one column of
    // the input.
    std::vector<TrainingEx> result;
    std::ifstream file(filename);
    while(true) {
        std::vector<double> in;
        for(int i=0; i != ninputs; ++i) {
            double x;
            file >> x;
            in.push_back(x*scaling_factors[i]);
        }
        double y;
        file >> y;
        if(file.eof())
            break;
        result.push_back(TrainingEx(in, y*scaling_factors[ninputs]));
    }
    return result;
}

int main() {
    // <3
    int ninputs = 2;
    std::vector<int> hidden_units = {1, 1};//31, 31, 31, 31};
    double learning_rate = 0.00003;
    auto nn = NeuralNet(ninputs, hidden_units, learning_rate);
    double scaling_factors[] = {1.0, 1.0, 1.0/10000};
    nn.train(getTrainingData(ninputs, "cylinder.train", scaling_factors));
    auto valid = nn.validate(getTrainingData(ninputs, "cylinder.validate",
                scaling_factors));
    printf("validation: %lf\n", valid);
    printf("Predicted volume of cylinder:%lf\n", nn.getOutput({1.0, 0.3}));
    return 0;
}
