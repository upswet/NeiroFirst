package ru.ext;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import lombok.SneakyThrows;
import lombok.experimental.FieldDefaults;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Функция активации
 */
interface FunActivationInterface extends Serializable {
    //вычислить функцию активации
    double calc(double x);

    //вычислить производную функции активации
    double deriative(double x);
}

class None implements FunActivationInterface {
    public double calc(double x) {
        return x;
    }

    public double deriative(double x) {
        return x;
    }

    public static None self = new None();
}

class Sigmoid implements FunActivationInterface {
    public double calc(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    public double deriative(double x) {
        return x * (1 - x);
    }

    public static Sigmoid self = new Sigmoid();
}

/**
 * Связь между нейронами
 */
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
class Link implements Serializable {
    @Setter(AccessLevel.NONE)
    double weight;//вес связи
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    double wd = 0F;//предыдущая корректировка веса данной связи (необходимо для использования момента).
    Neiron input; //входной нейрон (отправитель)
    Neiron output; //выходной нейрон (получатель)

    public Link(double weight, Neiron input, Neiron output) {
        this.weight = weight;
        this.input = input;
        this.output = output;

        this.input.getOutputs().add(this);
        this.output.getInputs().add(this);
    }

    /**
     * Передаёт сигнал получателю вызывая у того событие поулчения сигнала по одной из входящих связей
     *
     * @param value - передаваемый отправителем сигнал
     */
    public void sendSignal(double value) {
        output.onReciveInputSignal(value * weight);
    }

    /**
     * Корректировка веса связи на основе исходящего значения входного нейрона, дельты выходного и таких параметров как
     *
     * @param lr     - коэффициент обучения
     * @param moment - момент обучения (исп в методе момента который здесь применяется)
     */
    public void weightCorrect(double lr, double moment) {
        double grad = input.getValueOutput() * output.getDelta();
        wd = lr * grad + moment * wd;
        weight += wd;
    }

    /**
     * Разорвать данную связь
     */
    public void terminate() {
        this.input.getOutputs().remove(this);
        this.output.getInputs().remove(this);
    }
}

/**
 * Нейрон
 */
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
class Neiron implements Serializable {
    List<Link> inputs = new ArrayList<>(); //входные связи
    List<Link> outputs = new ArrayList<>(); //выходные связи
    FunActivationInterface fun; //функция активации
    @Setter
    double valueInput = 0.0;//входное значение
    double valueOutput;//выходное значение
    @Setter
    double delta; //дельта. Участвует в корректировке весов в алгоритме обратного распространения ошибки
    Boolean isBias = false;//это нейрон смещения?

    public Neiron(FunActivationInterface fun) {
        this.fun = fun;
    }

    public Neiron(FunActivationInterface fun, Boolean isBias) {
        this.fun = fun;
        this.isBias = isBias;
        if (isBias) {
            this.valueInput = 1.0;
            this.valueOutput = 1.0;
        }
    }

    /**
     * На основе суммарного полученного сигнала вычисляем сигнал который отправим дальше и рассылаем его всем выходным нейронам связанным с данным входным
     */
    public void sendOutputSignal() {
        valueOutput = fun.calc(valueInput);
        valueInput = 0.0;
        for (Link link : outputs)
            link.sendSignal(valueOutput);
    }

    /**
     * Событие получения сигнала нейроном по одной из его входящих связей
     *
     * @param value - полученный сигнал
     */
    public void onReciveInputSignal(double value) {
        valueInput += value;
    }
}

/**
 * Слой как совокупность нейронов
 */
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
class Layer implements Serializable {
    List<Neiron> neirons = new LinkedList<>();

    /**
     * Фабричный метод создания входного слоя
     *
     * @param count  - количество нейронов во входном слое
     * @param isBias - добавлять нейрон смещения или нет
     * @return - созданный слой
     */
    public static Layer createInputLayer(int count, boolean isBias) {
        Layer layer = new Layer();
        for (int i = 0; i < count; i++)
            layer.getNeirons().add(new Neiron(None.self));
        if (isBias)
            layer.getNeirons().add(new Neiron(None.self, true));
        return layer;
    }

    /**
     * Фабричный метод создания полносвязанного скрытого слоя
     *
     * @param count         - количество нейронов во входном слое
     * @param isBias        - добавлять нейрон смещения или нет
     * @param fun           - функция активации для нейроно слоя
     * @param previousLayer - предыдущий слой чтобы мы могли связать каждый его нейрон с нейроном создаваемого слоя
     * @return - созданный слой
     */
    public static Layer createHiddenLayer(int count, boolean isBias, FunActivationInterface fun, Layer previousLayer) {
        Layer layer = new Layer();
        for (int i = 0; i < count; i++) {
            Neiron neiron = new Neiron(fun);
            layer.getNeirons().add(neiron);
            //свяжем все нейроны предыдущего слоя с созданным нейроном сгенерировав связи со случайным весом
            for (Neiron previousNeiron : previousLayer.getNeirons()) {
                Link link = new Link(Net.random.nextDouble() - 0.5, previousNeiron, neiron);
            }
        }
        if (isBias)
            layer.getNeirons().add(new Neiron(None.self, true));
        return layer;
    }

    /**
     * Фабричный метод создания полносвязанного выходного слоя
     *
     * @param count         - количество нейронов во входном слое
     * @param fun           - функция активации для нейроно слоя
     * @param previousLayer - предыдущий слой чтобы мы могли связать каждый его нейрон с нейроном создаваемого слоя
     * @return - созданный слой
     */
    public static Layer createOutputLayer(int count, FunActivationInterface fun, Layer previousLayer) {
        return Layer.createHiddenLayer(count, false, fun, previousLayer);
    }
}

/**
 * Нейросеть как совокупность нейронов обрабатываемых и обучаемых послойно
 */
@FieldDefaults(level = AccessLevel.PRIVATE)
class Net implements Serializable {
    public static Random random = new Random();

    final List<Layer> layers;//слои нейросети
    final Neiron[] neirons; //все нейроны нейросети

    public Net(List<Layer> layers) {
        this.layers = layers;

        List<Neiron> neironList = new ArrayList<>(1000);
        for (Layer layer : layers)
            neironList.addAll(layer.getNeirons());

        this.neirons = new Neiron[neironList.size()];
        neironList.toArray(this.neirons);
    }

    /**
     * Прямое распространение сигнала
     */
    private void forwardPropagation() {
        for (Neiron neiron : neirons)
            neiron.sendOutputSignal();
    }

    /**
     * Обратное распространение ошибки и корректировка весов связей
     * @param target - целевой вектор данных
     * @param lr - скорость обучения
     * @param moment - момент обучения
     */
    private void backPropagation(double[] target, double lr, double moment){
        //Для нейронов входного слоя найдём дельту ошибки:
        List<Neiron> outputLayerNeirons= layers.getLast().getNeirons();
        for(int i=0; i<outputLayerNeirons.size(); i++) {
            Neiron outputLayerNeiron = outputLayerNeirons.get(i);
            //вычислим ошибку
            double err = target[i] - outputLayerNeiron.getValueOutput();
            //вычислим дельту ошибки и сохраним её в выходном нейроне
            outputLayerNeiron.setDelta(err * outputLayerNeiron.getFun().deriative(outputLayerNeiron.getValueOutput()));
        }

        //В обратном порядке для всех нейронов всех слоёв кроме выхоного найдём дельту ошибки и подкорректируем веса исходящих связей с учётом градиентов вычисленных по дельте ошибки нейронов-получателей в этих связях:
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer hiddenLayer = layers.get(i);
            for (Neiron neiron : hiddenLayer.getNeirons()) {
                //для тех нейронов у которых есть входные связи (т.е. которым нужно будет распр ошибку дальше) вычислим дельту ошибки
                //для всех нейронов обновим веса исходящих связей на основе переданного нейроном значения, веса связи и дельты ошибки нейрона-получателя данной связи
                double delta = 0;
                for (Link link : neiron.getOutputs()) {
                    if (!neiron.getInputs().isEmpty()) {
                        Neiron neiroOutput = link.getOutput();
                        delta += neiroOutput.getDelta() * link.getWeight();
                    }

                    //и сразу подкоректируем вес текущей связи
                    link.weightCorrect(lr, moment);
                }
                if (!neiron.getInputs().isEmpty())
                    neiron.setDelta(neiron.getFun().deriative(neiron.getValueOutput()) * delta);
            }
        }
    }

    /**
     * Получить значения из нейронов выходного слоя
     */
    private double[] getOutput() {
        Layer outputLayer = layers.getLast();
        double[] result = new double[outputLayer.getNeirons().size()];
        for (int i = 0; i < result.length; i++)
            result[i] = outputLayer.getNeirons().get(i).getValueOutput();
        return result;
    }

    /**
     * Поместить входные значения в нейроны входного слоя
     */
    private void setInput(double[] vector) {
        Layer inputLayer = layers.getFirst();
        for (int i = 0; i < vector.length; i++)
            inputLayer.getNeirons().get(i).setValueInput(vector[i]);
    }

    /**
     * Вычислить значения
     *
     * @param vector - вектор входных данных
     * @return - вектор выходных данных
     */
    public double[] calculate(double[] vector) {
        setInput(vector);
        forwardPropagation();
        return getOutput();
    }

    /**
     * Тренировка по одному вектору входных и выходных данных. Сначала прямой проход, потом обратное распространение ошибки
     *
     * @param data   - входные данные для нейронов входного слоя
     * @param target - ожидаемый правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr     - коэффициент обучения
     * @param moment -момент
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя
     */
    private double train(double[] data, double[] target, double lr, double moment) {
        double[] output = calculate(data);
        backPropagation(target, lr, moment);

        //вычислим ошибку на данном наборе данных
        double e = 0F;
        for (int i = 0; i < target.length; i++)
            e = Math.abs(target[i] - output[i]);
        return e * e;
    }

    /**
     * Тренировки по набору входных и выходных векторов данных.
     *
     * @param datas   - набор входных данные для нейронов входного слоя
     * @param targets - набор правильных выходных данных на нейронах выходного слоя. Обучение с учителем
     * @param lr      - коэффициент обучения
     * @param moment  -момент
     * @param epoch   - количество эпох. То есть полных повторов всех циклов обучения на переданных наборов входных и правильных выходных данных
     * @param minMse  - минимальная ошибка. При достижении обучение прекращается
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя
     */
    public double trains(List<double[]> datas, List<double[]> targets, double lr, double moment, Integer epoch, double minMse) {
        if (datas.size() != targets.size())
            throw new RuntimeException("Размерность множества данных для обучения не совпадает с размерностью множества ответов для обучения");

        double lastMse = 999;
        double mse = 999;

        while (epoch-- > 0) {
            Instant start = Instant.now();

            mse = 0.0;
            for (int i = 0; i < targets.size(); i++) {
                mse += train(datas.get(i), targets.get(i), lr, moment);
                if (i > 0 && i % 5000 == 0)
                    System.out.println("processed " + i + " from " + targets.size());
            }
            mse = mse / targets.size();
            System.out.println("epoch=" + epoch + ", mse=" + mse + ", duration(ms)=" + Duration.between(start, Instant.now()).toMillis());

            /*if (lastMse<mse){ //мы не смогли снизить ошибку на этой эпохе обучения. Подкоректируем гиперпараметры сети
                lr=lr/2;
                moment=moment/2;
                System.out.println("New set lr="+lr+", moment="+moment);
            }
            lastMse=mse;*/

            if (mse <= minMse) break;
        }
        return mse;
    }

    /**
     * Разорвать слабые связи, вес которых меньше заданного. Желательно после этого дополнительно обучить сеть
     */
    public int terminatWeakLink(double minWeigth) {
        int countTerminate = 0;
        for (Neiron neiron : neirons)
            for (Link link : new ArrayList<>(neiron.getOutputs()))
                if (Math.abs(link.getWeight()) < minWeigth) {
                    link.terminate();
                    countTerminate++;
                }
        System.out.println("terminatWeakLink count "+countTerminate);
        return countTerminate;
    }
}

/**
 * Сохраняет нейросеть в файл и загружает из него
 */
class FileSave {
    /**
     * Сохранить обученную нейросеть в файл по его пути
     *
     * @param filePath - путь к файлу в который сохраняем
     * @param object   - сохраняемый объект
     */
    public static void save(String filePath, Object object) {
        try {
            var fileOutput = new FileOutputStream(filePath);
            var objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(object);
            fileOutput.flush();
            objectOutput.flush();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("saving in " + filePath);
    }

    /**
     * Загрузить обученную нейросеть из файла
     *
     * @param filePath - путь к файлу из которого загружаем нейронку
     * @return - нейросеть
     */
    public static Object load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Object object = objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading from " + filePath);
            return object;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}

public class Base {
    public static void main(String[] args) throws URISyntaxException, IOException {
        //Net net=createNetForXor();
        //Net net = createNetForMnist();

        /*String fname="net_"+ LocalDateTime.now();
        FileSave.save(fname, net);
        net=(Net)FileSave.load(fname);*/
    }

    @SneakyThrows
    private static Net createNetForMnist() {
        List<double[]> datasTrain = new ArrayList<>();
        List<double[]> targetsTrain = new ArrayList<>();
        //prepareDataForMnist(targetsTrain, datasTrain, "mnist_train_100.csv");
        prepareDataForMnist(targetsTrain, datasTrain, "mnist_train.csv");
        List<double[]> datasTest = new ArrayList<>();
        List<double[]> targetsTest = new ArrayList<>();
        //prepareDataForMnist(targetsTest, datasTest, "mnist_test_10.csv");
        prepareDataForMnist(targetsTest, datasTest, "mnist_test.csv");

        //создаём нейросеть
        Layer inputLayer = Layer.createInputLayer(784, true);
        Layer hiddenLayer1 = Layer.createHiddenLayer(100, true, Sigmoid.self, inputLayer);
        Layer outputLayer = Layer.createOutputLayer(10, Sigmoid.self, hiddenLayer1);
        Net net = new Net(List.of(inputLayer, hiddenLayer1, outputLayer));

        //обучаем нейросеть
        double mse = net.trains(datasTrain, targetsTrain, 0.1, 0.01, 1, 0.001);
        //net.terminatWeakLink(0.01);
        //mse = net.trains(datasTrain, targetsTrain, 0.1, 0.01, 1, 0.01);


        System.out.println("check-net");
        Integer countError = 0;
        //прогоним через тестовый набор и посчитаем количество ошибок
        for (int i = 0; i < targetsTest.size(); i++) {
            double[] output = net.calculate(datasTest.get(i));
            Integer targetResult = findMax(targetsTest.get(i));
            Integer outputResult = findMax(output);
            if (targetResult != outputResult) {
                countError++;
                //System.out.println("targetResult="+targetResult+", outputResult="+outputResult);
            }
        }
        Integer len = targetsTest.size();
        System.out.println("eff=" + (float) (len - countError) / len);

        return net;
    }

    private static void prepareDataForMnist(List<double[]> targets, List<double[]> datas, String fileName) throws IOException, URISyntaxException {
        List<String> list = Files.readAllLines(Paths.get(Base.class.getClassLoader().getResource(fileName).toURI()));
        for (String s : list) {
            String[] sarr = s.split(",");

            double[] target = new double[10];
            Arrays.fill(target, 0.01);
            target[Integer.valueOf(sarr[0])] = 0.99;
            targets.add(target);

            double[] data = new double[sarr.length - 1];
            for (int j = 1; j < sarr.length; j++)
                data[j - 1] = (Double.valueOf(sarr[j]) / 255.0) * 0.99 + 0.01;

            datas.add(data);
        }
    }

    public static Integer findMax(double[] arr) {
        double max = -999F;
        Integer imax = -1;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > max) {
                max = arr[i];
                imax = i;
            }
        return imax;
    }

    private static Net createNetForXor() {
        //Обучим нейросеть вычислять XOR
        List<double[]> datasTrain = new ArrayList<>();
        datasTrain.add(new double[]{1F, 1F});
        datasTrain.add(new double[]{0F, 1F});
        datasTrain.add(new double[]{1F, 0F});
        datasTrain.add(new double[]{0F, 0F});
        List<double[]> targetsTrain = new ArrayList<>();
        targetsTrain.add(new double[]{0F});
        targetsTrain.add(new double[]{1F});
        targetsTrain.add(new double[]{1F});
        targetsTrain.add(new double[]{0F});

        //создаём нейросеть
        Layer inputLayer = Layer.createInputLayer(2, false);
        Layer hiddenLayer1 = Layer.createHiddenLayer(4, false, Sigmoid.self, inputLayer);
        Layer outputLayer = Layer.createOutputLayer(1, Sigmoid.self, hiddenLayer1);
        Net net = new Net(List.of(inputLayer, hiddenLayer1, outputLayer));

        //обучаем нейросеть
        double mse = net.trains(datasTrain, targetsTrain, 0.9, 0.03, 10000, 0.01);

        //Тестирование
        Integer countError = 0;
        //прогоним через тестовый набор и посчитаем количество ошибок
        for (int i = 0; i < targetsTrain.size(); i++) {
            double[] output = net.calculate(datasTrain.get(i));
            System.out.println("targetResult=" + targetsTrain.get(i)[0] + ", outputResult=" + output[0]);

            if (Math.abs(output[0] - targetsTrain.get(i)[0]) > 0.2)
                countError++;
        }
        Integer len = targetsTrain.size();
        float eff = (float) (len - countError) / len;
        System.out.println("eff=" + eff);

        return net;
    }
}
