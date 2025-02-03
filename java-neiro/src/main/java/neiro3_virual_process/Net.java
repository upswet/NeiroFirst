package neiro3_virual_process;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Set;

/**Описывает нейросеть как последовательность слоёв нейронов где каждый нейрон слоя N связан с каждым нейроном слоя N+1*/
public class Net implements Serializable {
    Layer inputLayer; //входной слой
    Layer outputLayer;//выходной слой
    Set<Neiro> biasNeiros;
    Neiro[] allNeiro;

    StatusNetEnum status = StatusNetEnum.READY;

    //параметры модели и результаты после цикла обучения
    volatile float lr;
    volatile float moment;
    Double mse;
    Integer epoch;

    volatile boolean threadRuning=true;

    public void start(int threadCount){
        /*int pool= allNeiro.length / threadCount;
        int start=0;
        int end=start+pool;
        for (int i=0; i<threadCount-1; i++){
            NetThread netThread = new NetThread(threadGroup,this, start, end);
            netThread.start();
            start=end;
            end=start+pool;
        }
        NetThread netThread = new NetThread(threadGroup,this, start, allNeiro.length);
        netThread.start();*/


        threadRuning=true;
        for (int i=0;i <allNeiro.length; i++){
            int finalI = i;
            Thread.ofVirtual().start(() -> performTask(this, finalI));
        }
        System.out.println("all thread is started");
    }

    private static void performTask(Net net, int index) {
        while (net.threadRuning){
            net.allNeiro[index].process(net.lr, net.moment);
            Thread.yield();
        }
    }

    public void stop(){
        // Прерывание всех потоков в группе
        threadRuning=false;
    }

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

    public Net(Layer inputLayer, Layer outputLayer, Neiro[] allNeiro, Set<Neiro> biasNeiros) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.allNeiro=allNeiro;
        this.biasNeiros=biasNeiros;
    }


    /**
     * Поместить входные данные в нейроны входного слоя
     * @param data - данные
     */
    public void setData(float[] data) {
        int i=0;
        for (Neiro neiro : inputLayer.neiros) {
            neiro.setValue(data[i++]);
            neiro.status=StatusNeironEnum.READY;
            if (i>=data.length) break;
        }
        biasNeiros.forEach(neiro -> neiro.status=StatusNeironEnum.READY);
    }

    /**
     * Поместить идеальное целевое значение в нейроны выходного слоя для запуска процесса обр распр и корректировке весов связей
     * @param target - данные
     */
    public void setTarget(float[] target) {
        int i=0;
        for (Neiro neiro : outputLayer.neiros) {
            neiro.setTarget(target[i++]);
            if (i>=target.length) break;
        }
    }

    /**
     * Получить данные с нейронов выходного слоя
     *
     * @return - массив значений
     */
    public float[] getData() {
        float[] result = new float[this.outputLayer.neiros.size()];
        int i = 0;
        for (Neiro neiro : outputLayer.neiros)
            result[i++] = neiro.getValue();
        return result;
    }



    /**Прямое распространение сигнала от входного слоя к выходному
     * @param data - входные данные для нейронов входного слоя
     * @return - массив данных полученных на нейронах выходного слоя*/
    public float[] calculate(float[] data) {
        setData(data);

        /*for(int i=0;i<allNeiro.length;i++)
            if (!allNeiro[i].process(lr, moment)){
                new RuntimeException("rrrr");
            }*/
        this.outputLayer.countNeiroOnForward.set(0);
        while (!this.outputLayer.isOnReadyForward()){
            Thread.yield();
        }

        return getData();
    }

    /**Сбросить статус нейронов входного слоя чтобы можно было гонять прямой проход за прямым проходом в режиме эксплуатации (а не обучения) нейросети*/
    public void resetNeironStatus(){
        for (Neiro neiro : allNeiro) neiro.status = StatusNeironEnum.READY;
    }

    /**Обратное респространение ошибки от нейронов выходного слоя к нейронам входного слоя с коррекцией весов связей участвующих в процессе
     * @param target - правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент*/
    private void backward(float[] target, float lr, float moment) {
        setTarget(target);

        /*for(int i=allNeiro.length-1;i>=0;i--)
            if (!allNeiro[i].process(lr, moment)){
                new RuntimeException("rrrr");
            }*/
        this.inputLayer.countNeiroOnBackward.set(0);
        while (!this.inputLayer.isOnReadyBackward()){
            Thread.yield();
        }
    }

    /**Цикл тренировки по одному вектору входных и выходных данных. Сначала прямой проход, потом обратное распространение ошибки
     * @param data - входные данные для нейронов входного слоя
     * @param target - ожидаемый правильный выход нейронов выходного слоя. Обучение с учителем
     * @param lr - коэффициент обучения
     * @param moment -момент
     * @return - квадрат ошибки суммированной по всем нейронам выходного слоя*/
    public float train(float[] data, float[] target, float lr, float moment){
        float[] output=calculate(data);
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
                if (i>0 && i%10==0)
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
