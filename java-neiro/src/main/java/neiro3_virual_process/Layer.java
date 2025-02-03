package neiro3_virual_process;

import java.io.Serializable;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**Описывает слой как совокупность нейронов с которыми связаны нейроны предыдущего слоя и которые сами связаны с нейронами следующего слоя*/
public class Layer implements Serializable {

    Set<Neiro> neiros = new LinkedHashSet<>(); //совокупность нейронов
    AtomicInteger countNeiroOnForward =new AtomicInteger(0);//только для выходного слоя: кол нейронов из слоя завершивших процесс прямого распространения сигнала
    AtomicInteger countNeiroOnBackward=new AtomicInteger(0);//только для выходного слоя: кол нейронов из слоя завершивших процесс обратного распространения ошибки

    Boolean isBias=false;

    public Layer() { }

    /**Все нейроны этого слоя закончили прямое распространение?*/
    public boolean isOnReadyForward(){
        return neiros.size() == countNeiroOnForward.get() + (isBias ? 1 : 0);
    }
    /**Все нейроны этого слоя закончили обратное распространение?*/
    public boolean isOnReadyBackward(){
        return neiros.size() == countNeiroOnBackward.get() + (isBias ? 1 : 0);
    }

    /**Добавить к слою нейрон смещения
     * @return - добавленный нейрон*/
    public Neiro addBias(){
        Neiro neiroBias = new Neiro(TypeEnum.BIAS, FunEnum.NONE);
        this.neiros.add(neiroBias);
        this.isBias=true;
        return neiroBias;
    }

    /**
     * Создание входного слоя
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @return - созданный слой
     */
    public static Layer createLayerInput(Integer neiroCount, FunEnum fun) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++)
            layer.neiros.add(new Neiro(TypeEnum.INPUT, fun, layer));

        return layer;
    }

    /**
     * Создание полносвязанного (т.е. каждый нейрон создаваемого слоя будет связан с каждым нейроном предыдущего слоя) скрытого слоя
     * @param previousLayer - предыдущий слой
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @return - созданный слой
     */
    public static Layer createLayerHidden(Integer neiroCount, FunEnum fun, Layer previousLayer) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++) {
            Neiro neiro = new Neiro(TypeEnum.HIDDEN, fun);
            layer.neiros.add(neiro);
            //создание связей каждого нейрона предыдщего слоя с только что созданным
            for (Neiro neiroForPreviousLayer : previousLayer.neiros) {
                Link link = new Link(neiroForPreviousLayer, neiro, Link.generateWeight());
            }
        }

        return layer;
    }

    /**
     * Создание полносвязанного (т.е. каждый нейрон создаваемого слоя будет связан с каждым нейроном предыдущего слоя) выходного слоя
     * @param previousLayer - предыдущий слой
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @return - созданный слой
     */
    public static Layer createLayerOutput(Integer neiroCount, FunEnum fun, Layer previousLayer) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++) {
            Neiro neiro = new Neiro(TypeEnum.OUTPUT, fun, layer);
            layer.neiros.add(neiro);
            //создание связей каждого нейрона предыдщего слоя с только что созданным
            for (Neiro neiroForPreviousLayer : previousLayer.neiros) {
                Link link = new Link(neiroForPreviousLayer, neiro, Link.generateWeight());
            }
        }

        return layer;
    }


    public void neiroOnFrward(){
        countNeiroOnForward.incrementAndGet();
    }

    public void neiroOnBackward(){
        countNeiroOnBackward.incrementAndGet();
    }
}
