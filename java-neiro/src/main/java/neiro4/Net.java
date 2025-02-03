package neiro4;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Set;

/**Описывает нейросеть как упорядоченную совокупность нейронов*/
public class Net implements Serializable {
    Neiro[] neiros;

    //параметры модели и результаты после цикла обучения
    float lr;
    float moment;
    Double mse;
    Integer epoch;

    /**Сохранить обученную нейросеть в файл по его пути
     * @param filePath - путь к файлу в который сохраняем
     * @param net - сохраняемая нейросеть
     * @param eff - эффективность по результатам тестирования*/
    public static void save(String filePath, Net net, float eff) {
        filePath+="_"+"e"+net.epoch+"_"+net.mse+"_"+eff;
        try {
            var fileOutput = new FileOutputStream(filePath);
            var objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(net);
            fileOutput.flush();
            objectOutput.flush();
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        System.out.println("saving net in "+filePath);
    }
    /**Загрузить обученную нейросеть из файла
     * @param filePath - путь к файлу из которого загружаем нейронку
     * @return - нейросеть*/
    public static Net load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Net net = (Net) objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading net from "+filePath);
            return net;
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    public Net(List<Neiro> neiroList) {
        this.neiros=neiroList.toArray(new Neiro[0]);
    }


    /**
     * Поместить входные данные в нейроны входного слоя
     * @param data - данные
     */
    public void setData(float[] data) {
        for (int i=0; i<data.length; i++)
            neiros[i].setValue(data[i]);
    }

    /**
     * Поместить идеальное целевое значение в нейроны выходного слоя для запуска процесса обр распр и корректировке весов связей
     * @param target - данные
     */
    public void setTarget(float[] target) {
        for (int i=0; i<target.length; i++)
            neiros[neiros.length-target.length+i].setTarget(target[i]);
    }

    /**
     * Получить данные с нейронов выходного слоя
     * @param outputNeiroCount - количество нейронов в выходном слое
     * @return - массив значений
     */
    public float[] getData(int outputNeiroCount) {
        float[] result = new float[outputNeiroCount];
        for (int i=0; i<outputNeiroCount; i++)
            result[i] = neiros[neiros.length-outputNeiroCount+i].getValue();
        return result;
    }



    /**Прямое распространение сигнала от входного слоя к выходному
     * @param data - входные данные для нейронов входного слоя
     * @param outputNeiroCount - количество нейронов в выходном слое
     * @return - массив данных полученных на нейронах выходного слоя*/
    public float[] calculate(float[] data, int outputNeiroCount) {
        setData(data);

        for(int i=0; i<neiros.length; i++)
            neiros[i].forwardProcess();

        return getData(outputNeiroCount);
    }

    /**Обратное респространение ошибки от нейронов выходного слоя к нейронам входного слоя с коррекцией весов связей участвующих в процессе
     * @param target - правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент*/
    private void backward(float[] target, float lr, float moment) {
        setTarget(target);

        for(int i=neiros.length-1; i>=0; i--)
            neiros[i].backwardProcess(lr, moment);
    }

    /**Цикл тренировки по одному вектору входных и выходных данных. Сначала прямой проход, потом обратное распространение ошибки
     * @param data - входные данные для нейронов входного слоя
     * @param target - ожидаемый правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя*/
    public float train(float[] data, float[] target, float lr, float moment){
        float[] output=calculate(data, target.length);
        backward(target, lr, moment);

        //вычислим ошибку на данном наборе данных
        float e=0F;
        for (int i=0;i<target.length;i++)
            e = Math.abs(target[i] - output[i]);
        return e*e;
    }


    /**Циклы тренировки по набору входных и выходных данных.
     * @param datas - набор входных данные для нейронов входного слоя
     * @param targets - набор правильных выходных данных на нейронах выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент
     * @param epoch - количество эпох. То есть полных повторов всех циклов обучения на переданных наборов входных и правильных выходных данных
     * @param minMse - минимальная ошибка. При достижении обучение прекращается
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя*/
    public void trains(List<float[]> datas, List<float[]> targets, float lr, float moment, Integer epoch, float minMse){
        if (datas.size()!=targets.size())
            throw new RuntimeException("Размерность множества данных для обучения не совпадает с размерностью множества ответов для обучения");

        this.lr=lr;
        this.moment=moment;
        this.epoch=epoch;
        this.mse=999.0;

        while (epoch-->0) {
            Instant start = Instant.now();

            Double mse = 0.0;
            for (int i = 0; i < targets.size(); i++) {
                mse += train(datas.get(i), targets.get(i), this.lr, this.moment);
                if (i>0 && i%5000==0)
                    System.out.println("processed "+i+" from "+targets.size());
            }
            mse=mse / targets.size();
            System.out.println("epoch="+epoch+", mse="+mse+", duration(ms)="+Duration.between(start, Instant.now()).toMillis());

            if (this.mse<mse){ //мы не смогли снизить ошибку на этой эпохе обучения. Подкоректируем гиперпараметры сети
                this.lr=this.lr/2;
                this.moment=this.moment/2;
                System.out.println("New set lr="+this.lr+", moment="+this.moment);
            }
            this.mse=mse;

            if (mse<=minMse) break;
        }
    }
}
