#include <vector>
#include <cmath>
#include <algorithm>

#include "avisynth.h"

typedef double FLType;
const FLType FLType_MAX = sizeof(FLType) < 8 ? FLT_MAX : DBL_MAX;
const FLType FLType_MIN = sizeof(FLType) < 8 ? FLT_MIN : DBL_MIN;

AVS_FORCEINLINE void* aligned_malloc(size_t size, size_t align)
{
    void* result = [&]() {
#ifdef _MSC_VER 
        return _aligned_malloc(size, align);
#else 
        if (posix_memalign(&result, align, size))
            return result = nullptr;
        else
            return result;
#endif
    }();

    return result;
}

AVS_FORCEINLINE void aligned_free(void* ptr)
{
#ifdef _MSC_VER 
    _aligned_free(ptr);
#else 
    free(ptr);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


constexpr double Pi = 3.1415926535897932384626433832795;
const double sqrt_2Pi = sqrt(2 * Pi);

constexpr double sigmaSMul = 2.;
const double sigmaRMul = sizeof(FLType) < 8 ? 8. : 32.; // 8. when FLType is float, 32. when FLType is double

AVS_FORCEINLINE double Gaussian_Function(double x, double sigma)
{
    x /= sigma;
    return exp(x * x / -2);
}

AVS_FORCEINLINE double Gaussian_Function_sqr_x(double sqr_x, double sigma)
{
    return exp(sqr_x / (sigma * sigma * -2));
}

AVS_FORCEINLINE double Normalized_Gaussian_Function(double x, double sigma)
{
    x /= sigma;
    return exp(x * x / -2) / (sqrt_2Pi * sigma);
}

AVS_FORCEINLINE double Normalized_Gaussian_Function_sqr_x(double sqr_x, double sigma)
{
    return exp(sqr_x / (sigma * sigma * -2)) / (sqrt_2Pi * sigma);
}

AVS_FORCEINLINE FLType* Gaussian_Function_Spatial_LUT_Generation(const int xUpper, const int yUpper, const double sigmaS)
{
    int x, y;
    FLType* GS_LUT = new FLType[xUpper * yUpper];

    for (y = 0; y < yUpper; ++y)
    {
        for (x = 0; x < xUpper; ++x)
            GS_LUT[y * xUpper + x] = static_cast<FLType>(Gaussian_Function_sqr_x(static_cast<FLType>(x * x + y * y), sigmaS));
    }

    return GS_LUT;
}

AVS_FORCEINLINE FLType Gaussian_Distribution2D_Spatial_LUT_Lookup(const FLType* GS_LUT, const int xUpper, const int x, const int y)
{
    return GS_LUT[y * xUpper + x];
}

AVS_FORCEINLINE void Gaussian_Function_Spatial_LUT_Free(FLType* GS_LUT)
{
    delete[] GS_LUT;
}

AVS_FORCEINLINE FLType* Gaussian_Function_Range_LUT_Generation(const int ValueRange, double sigmaR)
{
    int i;
    int Levels = ValueRange + 1;
    const int upper = std::min(ValueRange, static_cast<int>(sigmaR * sigmaRMul * ValueRange + 0.5));
    FLType* GR_LUT = new FLType[Levels];

    for (i = 0; i <= upper; ++i)
        GR_LUT[i] = static_cast<FLType>(Normalized_Gaussian_Function(static_cast<double>(i) / ValueRange, sigmaR));
    // For unknown reason, when more range weights are too small or equal 0, the runtime speed gets lower - mainly in function Recursive_Gaussian2D_Horizontal.
    // To avoid this issue, we set range weights whose range values are larger than sigmaR*sigmaRMul to the Gaussian function value at sigmaR*sigmaRMul.
    if (i < Levels)
    {
        const FLType upperLUTvalue = GR_LUT[upper];

        for (; i < Levels; ++i)
            GR_LUT[i] = upperLUTvalue;
    }

    return GR_LUT;
}

template < typename T >
AVS_FORCEINLINE FLType Gaussian_Distribution2D_Range_LUT_Lookup(const FLType* GR_LUT, const T Value1, const T Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}

AVS_FORCEINLINE void Gaussian_Function_Range_LUT_Free(FLType* GR_LUT)
{
    delete[] GR_LUT;
}

void Recursive_Gaussian_Parameters(const double sigma, FLType& B, FLType& B1, FLType& B2, FLType& B3)
{
    const double q = sigma < 2.5 ? 3.97156 - 4.14554 * sqrt(1 - 0.26891 * sigma) : 0.98711 * sigma - 0.96330;

    const double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    const double b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    const double b2 = -(1.4281 * q * q + 1.26661 * q * q * q);
    const double b3 = 0.422205 * q * q * q;

    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}

void Recursive_Gaussian2D_Vertical(FLType* output, const FLType* input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    int i0, i1, i2, i3, j, lower, upper;
    FLType P0, P1, P2, P3;

    if (output != input)
        memcpy(output, input, sizeof(FLType) * width);

    for (j = 0; j < height; ++j)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j < 1 ? i0 : i0 - stride;
        i2 = j < 2 ? i1 : i1 - stride;
        i3 = j < 3 ? i2 : i2 - stride;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = input[i0];
            output[i0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    }

    for (j = height - 1; j >= 0; --j)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j >= height - 1 ? i0 : i0 + stride;
        i2 = j >= height - 2 ? i1 : i1 + stride;
        i3 = j >= height - 3 ? i2 : i2 + stride;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = output[i0];
            output[i0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    }
}

void Recursive_Gaussian2D_Horizontal(FLType* output, const FLType* input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    int i, j, lower, upper;
    FLType P0, P1, P2, P3;

    for (j = 0; j < height; ++j)
    {
        lower = stride * j;
        upper = lower + width;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (++i; i < upper; ++i)
        {
            P0 = B * input[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        --i;
        P3 = P2 = P1 = output[i];

        for (--i; i >= lower; --i)
        {
            P0 = B * output[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Multi Scale Retinex process kernel for floating point data
void MSRKernel(FLType* odata, const FLType* idata, const int pcount, const int stride, const int width, const int height, const int sigma_size, const std::vector<double>& sigma)
{
    //FLType FloorFL = 0;
    //FLType CeilFL = 1;

    FLType* gauss = reinterpret_cast<FLType*>(aligned_malloc(sizeof(FLType) * pcount, 32));

    for (int j = 0; j < height; j++)
    {
        int i = stride * j;

        for (int upper = i + width; i < upper; i++)
            odata[i] = 1;
    }

    FLType B, B1, B2, B3;

    for (int s = 0; s < sigma_size; s++)
    {
        if (sigma[s] > 0)
        {
            Recursive_Gaussian_Parameters(sigma[s], B, B1, B2, B3);
            Recursive_Gaussian2D_Horizontal(gauss, idata, height, width, stride, B, B1, B2, B3);
            Recursive_Gaussian2D_Vertical(gauss, gauss, height, width, stride, B, B1, B2, B3);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    odata[i] *= gauss[i] <= 0 ? 1 : idata[i] / gauss[i] + 1;
            }
        }
        else
        {
            for (int j = 0; j < height; j++)
            {
                int i = stride * j;
                for (int upper = i + width; i < upper; i++)
                    odata[i] *= FLType(2);
            }
        }
    }

    for (int j = 0; j < height; j++)
    {
        int i = stride * j;

        for (int upper = i + width; i < upper; i++)
            odata[i] = log(odata[i]) / static_cast<FLType>(sigma_size);
    }

    aligned_free(gauss);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Common : public GenericVideoFilter
{
protected:
    std::vector<double> sigma;
    double lower_thr;
    double upper_thr;
    const int HistBins = 4096;
    bool fulls;
    bool fulld;
    bool v8 = true;

public:
    Common(PClip _child, const std::vector<double>& i_sigma, const double i_lower_thr, const double i_upper_thr, const bool i_fulls, const bool i_fulld, IScriptEnvironment* env)
        : GenericVideoFilter(_child), sigma(i_sigma), lower_thr(i_lower_thr), upper_thr(i_upper_thr), fulls(i_fulls), fulld(i_fulld)
    {
        if (vi.ComponentSize() > 2 || !vi.IsPlanar() || (!vi.IsRGB() && !vi.Is444() && !vi.IsY()))
            env->ThrowError("The inplut clip must be in Y/YUV444/RGB 8..16-bit planar format.");

        for (unsigned i = 0; i < sigma.size(); ++i)
        {
            if (sigma[i] < 0.0)
                env->ThrowError("sigma must be non-negative float number.");
        }

        if (lower_thr < 0.0 || lower_thr >= 1.0)
            env->ThrowError("lower_thr must be greater than 0.0 and lower than 1.0.");
        if (upper_thr < 0.0 || upper_thr >= 1.0)
            env->ThrowError("upper_thr must be greater than 0.0 and lower than 1.0.");
        if (lower_thr + upper_thr >= 1.0)
            env->ThrowError("The sum of lower_thr and upper_thr must be lower than 1.0");

        try { env->CheckVersion(8); }
        catch (const AvisynthError&) { v8 = false; };
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class MSRCP : public Common
{
    double chroma_protect;

    // Simplest color balance with pixel clipping on either side of the dynamic range
    void SimplestColorBalance(FLType* odata, const FLType* idata, const int pcount, const int stride, const int width, const int height); // odata as input and output, idata as source

    template <typename T>
    void process_core(PVideoFrame& dst, PVideoFrame& src);
    void (MSRCP::* process)(PVideoFrame& dst, PVideoFrame& src);

public:
    MSRCP(PClip _child, const std::vector<double>& i_sigma, const double i_lower_thr, const double i_upper_thr, const bool i_fulls, const bool i_fulld, const double i_chroma_protect, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


void MSRCP::SimplestColorBalance(FLType* odata, const FLType* idata, const int pcount, const int stride, const int width, const int height)
{
    FLType offset, gain;
    FLType min = FLType_MAX;
    FLType max = -FLType_MAX;

    const FLType FloorFL = 0;
    const FLType CeilFL = 1;

    for (int j = 0; j < height; j++)
    {
        int i = stride * j;

        for (int upper = i + width; i < upper; i++)
        {
            min = std::min(min, odata[i]);
            max = std::max(max, odata[i]);
        }
    }

    if (max <= min)
    {
        memcpy(odata, idata, sizeof(FLType) * pcount);
        return;
    }

    if (lower_thr > 0 || upper_thr > 0)
    {
        int Count, MaxCount;

        int* Histogram = reinterpret_cast<int*>(aligned_malloc(sizeof(int) * HistBins, 32));
        memset(Histogram, 0, sizeof(int) * HistBins);

        gain = (HistBins - 1) / (max - min);
        offset = -min * gain;

        for (int j = 0; j < height; j++)
        {
            int i = stride * j;

            for (int upper = i + width; i < upper; i++)
                Histogram[static_cast<int>(odata[i] * gain + offset)]++;
        }

        gain = (max - min) / (HistBins - 1);
        offset = min;

        Count = 0;
        MaxCount = static_cast<int>(width * height * lower_thr + 0.5);

        int h;

        for (h = 0; h < HistBins; h++)
        {
            Count += Histogram[h];
            if (Count > MaxCount) break;
        }

        min = h * gain + offset;

        Count = 0;
        MaxCount = static_cast<int>(width * height * upper_thr + 0.5);

        for (h = HistBins - 1; h >= 0; h--)
        {
            Count += Histogram[h];
            if (Count > MaxCount) break;
        }

        max = h * gain + offset;

        aligned_free(Histogram);
    }

    gain = (CeilFL - FloorFL) / (max - min);
    offset = -min * gain + FloorFL;

    if (lower_thr > 0 || upper_thr > 0)
    {
        for (int j = 0; j < height; j++)
        {
            int i = stride * j;

            for (int upper = i + width; i < upper; i++)
                odata[i] = std::clamp(odata[i] * gain + offset, FloorFL, CeilFL);
        }
    }
    else
    {
        for (int j = 0; j < height; j++)
        {
            int i = stride * j;
            for (int upper = i + width; i < upper; i++)
                odata[i] = odata[i] * gain + offset;
        }
    }
}

template <typename T>
void MSRCP::process_core(PVideoFrame& dst, PVideoFrame& src)
{
    const int bps = vi.BitsPerComponent();

    const int stride = src->GetPitch() / sizeof(T);
    const int width = src->GetRowSize() / sizeof(T);
    const int height = src->GetHeight();
    const int pcount = stride * height;

    // Calculate quantization parameters according to bit per sample and limited/full range
    // Floor and Ceil for limited range src will be determined later according to minimum and maximum value in the frame
    T sFloor = 0;
    //T sFloorC = 0;
    const int sNeutral = 128 << (bps - 8);
    T sCeil = (1 << bps) - 1;
    //T sCeilC = (1 << bps) - 1;
    const T sRange = fulls ? (1 << bps) - 1 : 219 << (bps - 8);
    const T sRangeC = fulls ? (1 << bps) - 1 : 224 << (bps - 8);
    const T dFloor = fulld ? 0 : 16 << (bps - 8);
    //T dFloorC = d.fulld ? 0 : 16 << (bps - 8);
    const int dNeutral = 128 << (bps - 8);
    //T dCeil = d.fulld ? (1 << bps) - 1 : 235 << (bps - 8);
    //T dCeilC = d.fulld ? (1 << bps) - 1 : 240 << (bps - 8);
    const T dRange = fulld ? (1 << bps) - 1 : 219 << (bps - 8);
    const T dRangeC = fulld ? (1 << bps) - 1 : 224 << (bps - 8);
    FLType sFloorFL = static_cast<FLType>(sFloor);
    //FLType sFloorCFL = static_cast<FLType>(sFloorC);
    //FLType sNeutralFL = static_cast<FLType>(sNeutral);
    //FLType sCeilFL = static_cast<FLType>(sCeil);
    //FLType sCeilCFL = static_cast<FLType>(sCeilC);
    const FLType sRangeFL = static_cast<FLType>(sRange);
    const FLType sRangeCFL = static_cast<FLType>(sRangeC);
    const  FLType sRangeC2FL = static_cast<FLType>(sRangeC) / 2.;
    const FLType dFloorFL = static_cast<FLType>(dFloor);
    //FLType dFloorCFL = static_cast<FLType>(dFloorC);
    const FLType dNeutralFL = static_cast<FLType>(dNeutral);
    //FLType dCeilFL = static_cast<FLType>(dCeil);
    //FLType dCeilCFL = static_cast<FLType>(dCeilC);
    const FLType dRangeFL = static_cast<FLType>(dRange);
    const FLType dRangeCFL = static_cast<FLType>(dRangeC);

    // Allocate floating point data buff
    FLType* idata = reinterpret_cast<FLType*>(aligned_malloc(sizeof(FLType) * pcount, 32));
    FLType* odata = reinterpret_cast<FLType*>(aligned_malloc(sizeof(FLType) * pcount, 32));

    if (vi.IsY()) // Procedure for Gray color family
    {
        // Get read and write pointer for src and dst
        const T* Ysrcp = reinterpret_cast<const T*>(src->GetReadPtr());
        T* Ydstp = reinterpret_cast<T*>(dst->GetWritePtr());

        // Derive floating point intensity channel from integer Y channel
        if (fulls)
        {
            const FLType gain = 1 / sRangeFL;

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = Ysrcp[i] * gain;
            }
        }
        else
        {
            // If src is of limited range, determine the Floor and Ceil by the minimum and maximum value in the frame
            T min = sCeil;
            T max = sFloor;

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                {
                    min = std::min(min, Ysrcp[i]);
                    max = std::max(max, Ysrcp[i]);
                }
            }
            if (max > min)
            {
                sFloor = min;
                sCeil = max;
                sFloorFL = static_cast<FLType>(sFloor);
                //sCeilFL = static_cast<FLType>(sCeil);
            }

            FLType gain = 1 / static_cast<FLType>(sCeil - sFloor);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = (Ysrcp[i] - sFloor) * gain;
            }
        }

        // Apply MSR to floating point intensity channel
        MSRKernel(odata, idata, pcount, stride, width, height, sigma.size(), sigma);
        // Simplest color balance with pixel clipping on either side of the dynamic range
        SimplestColorBalance(odata, idata, pcount, stride, width, height);

        // Convert floating point intensity channel to integer Y channel
        const FLType offset = dFloorFL + FLType(0.5);

        for (int j = 0; j < height; j++)
        {
            int i = stride * j;

            for (int upper = i + width; i < upper; i++)
                Ydstp[i] = static_cast<T>(odata[i] * dRangeFL + offset);
        }
    }
    else if (vi.IsRGB()) // Procedure for RGB color family
    {
        // Get read and write pointer for src and dst
        const T* Rsrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_R));
        const T* Gsrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_G));
        const T* Bsrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_B));
        T* Rdstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_R));
        T* Gdstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_G));
        T* Bdstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_B));

        // Derive floating point intensity channel from integer RGB channel
        if (fulls)
        {
            const FLType gain = 1 / (sRangeFL * 3);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = (Rsrcp[i] + Gsrcp[i] + Bsrcp[i]) * gain;
            }
        }
        else
        {
            // If src is of limited range, determine the Floor and Ceil by the minimum and maximum value in the frame
            T min = sCeil;
            T max = sFloor;

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                {
                    min = std::min(min, std::min(Rsrcp[i], std::min(Gsrcp[i], Bsrcp[i])));
                    max = std::max(max, std::max(Rsrcp[i], std::max(Gsrcp[i], Bsrcp[i])));
                }
            }
            if (max > min)
            {
                sFloor = min;
                sCeil = max;
                sFloorFL = static_cast<FLType>(sFloor);
                //sCeilFL = static_cast<FLType>(sCeil);
            }

            const FLType offset = sFloorFL * -3;
            const FLType gain = 1 / (static_cast<FLType>(sCeil - sFloor) * 3);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = (Rsrcp[i] + Gsrcp[i] + Bsrcp[i] + offset) * gain;
            }
        }

        // Apply MSR to floating point intensity channel
        MSRKernel(odata, idata, pcount, stride, width, height, sigma.size(), sigma);
        // Simplest color balance with pixel clipping on either side of the dynamic range
        SimplestColorBalance(odata, idata, pcount, stride, width, height);

        // Adjust integer RGB channel according to filtering result in floating point intensity channel
        //T Rval, Gval, Bval;

        if (sFloor == 0 && dFloorFL == 0 && sRangeFL == dRangeFL)
        {
            const FLType offset = FLType(0.5);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                {
                    const T Rval = Rsrcp[i];
                    const T Gval = Gsrcp[i];
                    const T Bval = Bsrcp[i];
                    FLType gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                    gain = std::min(sRangeFL / std::max(Rval, std::max(Gval, Bval)), gain);
                    Rdstp[i] = static_cast<T>(Rval * gain + offset);
                    Gdstp[i] = static_cast<T>(Gval * gain + offset);
                    Bdstp[i] = static_cast<T>(Bval * gain + offset);
                }
            }
        }
        else
        {
            const FLType scale = dRangeFL / sRangeFL;
            const FLType offset = dFloorFL + FLType(0.5);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                {
                    const T Rval = Rsrcp[i] - sFloor;
                    const T Gval = Gsrcp[i] - sFloor;
                    const T Bval = Bsrcp[i] - sFloor;
                    FLType gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                    gain = std::min(sRangeFL / std::max(Rval, std::max(Gval, Bval)), gain) * scale;
                    Rdstp[i] = static_cast<T>(Rval * gain + offset);
                    Gdstp[i] = static_cast<T>(Gval * gain + offset);
                    Bdstp[i] = static_cast<T>(Bval * gain + offset);
                }
            }
        }
    }
    else // Procedure for YUV color family
    {
        // Get read and write pointer for src and dst
        const T* Ysrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_Y));
        const T* Usrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_U));
        const T* Vsrcp = reinterpret_cast<const T*>(src->GetReadPtr(PLANAR_V));
        T* Ydstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_Y));
        T* Udstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_U));
        T* Vdstp = reinterpret_cast<T*>(dst->GetWritePtr(PLANAR_V));

        // Derive floating point intensity channel from integer Y channel
        if (fulls)
        {
            const FLType gain = 1 / sRangeFL;

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = Ysrcp[i] * gain;
            }
        }
        else
        {
            // If src is of limited range, determine the Floor and Ceil by the minimum and maximum value in the frame
            T min = sCeil;
            T max = sFloor;

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                {
                    min = std::min(min, Ysrcp[i]);
                    max = std::max(max, Ysrcp[i]);
                }
            }
            if (max > min)
            {
                sFloor = min;
                sCeil = max;
                sFloorFL = static_cast<FLType>(sFloor);
                //sCeilFL = static_cast<FLType>(sCeil);
            }

            const FLType gain = 1 / static_cast<FLType>(sCeil - sFloor);

            for (int j = 0; j < height; j++)
            {
                int i = stride * j;

                for (int upper = i + width; i < upper; i++)
                    idata[i] = (Ysrcp[i] - sFloor) * gain;
            }
        }

        // Apply MSR to floating point intensity channel
        MSRKernel(odata, idata, pcount, stride, width, height, sigma.size(), sigma);
        // Simplest color balance with pixel clipping on either side of the dynamic range
        SimplestColorBalance(odata, idata, pcount, stride, width, height);

        // Convert floating point intensity channel to integer Y channel
        // Adjust integer UV channel according to filtering result in floating point intensity channel
        // Chroma protect uses log function to attenuate the adjustment in UV channel
        const FLType chroma_protect_mul1 = static_cast<FLType>(chroma_protect - 1);
        const FLType chroma_protect_mul2 = static_cast<FLType>(1 / log(chroma_protect));

        const FLType scale = dRangeCFL / sRangeCFL;
        const FLType    offset = (fulld) ? (dNeutralFL + FLType(0.499999)) : (dNeutralFL + FLType(0.5));
        const FLType offsetY = dFloorFL + FLType(0.5);

        for (int j = 0; j < height; j++)
        {
            int i = stride * j;

            for (int upper = i + width; i < upper; i++)
            {
                const int Uval = Usrcp[i] - sNeutral;
                const int Vval = Vsrcp[i] - sNeutral;

                FLType gain = [&]() {
                    if (chroma_protect > 1)
                        return idata[i] <= 0 ? 1 : log(odata[i] / idata[i] * chroma_protect_mul1 + 1) * chroma_protect_mul2;
                    else
                        return idata[i] <= 0 ? 1 : odata[i] / idata[i];
                }();

                gain = (dRangeCFL == sRangeCFL) ? (std::min(sRangeC2FL / std::max(std::abs(Uval), std::abs(Vval)), gain)) : (std::min(sRangeC2FL / std::max(std::abs(Uval), std::abs(Vval)), gain) * scale);

                Ydstp[i] = static_cast<T>(odata[i] * dRangeFL + offsetY);
                Udstp[i] = static_cast<T>(Uval * gain + offset);
                Vdstp[i] = static_cast<T>(Vval * gain + offset);
            }
        }
    }

    // Free floating point data buff
    aligned_free(idata);
    aligned_free(odata);
}

MSRCP::MSRCP(PClip _child, const std::vector<double>& i_sigma, const double i_lower_thr, const double i_upper_thr, const bool i_fulls, const bool i_fulld, const double i_chroma_protect, IScriptEnvironment* env)
    : Common(_child, i_sigma, i_lower_thr, i_upper_thr, i_fulls, i_fulld, env), chroma_protect(i_chroma_protect)
{
    if (chroma_protect < 1)
        env->ThrowError("MSRCP: chroma_protect must be equal to or greater than 1.0.");

    process = (vi.ComponentSize() == 1) ? &MSRCP::process_core<uint8_t> : &MSRCP::process_core<uint16_t>;
}

PVideoFrame __stdcall MSRCP::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = (v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    (this->*process)(dst, src);

    return dst;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


AVSValue __cdecl Create_MSRCP(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    std::vector<double> sigma;

    if (args[1].ArraySize())
    {
        if (args[1].IsArray())
        {
            for (int i = 0; i < args[1].ArraySize(); ++i)
                sigma.emplace_back(args[1][i].AsFloatf());
        }
        else
            sigma.emplace_back(args[1].AsFloatf());
    }
    else
        sigma = { 25.0, 80.0, 250.0 };

    PClip clip = args[0].AsClip();
    const bool fulls = args[4].AsBool((clip->GetVideoInfo().IsRGB()) ? true : false);

    return new MSRCP(clip,
        sigma,
        args[2].AsFloatf(0.001f),
        args[3].AsFloatf(0.001f),
        fulls,
        args[5].AsBool(fulls),
        args[6].AsFloatf(1.2f),
        env);
}

/*
AVSValue __cdecl Create_MSRCR(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    std::vector<double> sigma;
    if (args[1].Defined())
    {
        for (int i = 0; i < args[1].ArraySize(); ++i)
            sigma.emplace_back(args[1][i].AsFloatf());
    }
    else
        sigma = { 25.0, 80.0, 250.0 };

    const bool fulls = args[4].AsBool(true);

    return new MSRCR(args[0].AsClip(),
        sigma,
        args[2].AsFloatf(0.001f),
        args[3].AsFloatf(0.001f),
        fulls,
        args[5].AsBool(fulls),
        args[6].AsFloatf(125),
        env);
}
*/

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("MSRCP", "c[sigma]f*[lower_thr]f[upper_thr]f[fulls]b[fulld]b[chroma_protect]f", Create_MSRCP, 0);
    //env->AddFunction("MSRCR", "c[sigma]f*[lower_thr]f[upper_thr]f[fulls]b[fulld]b[restore]f", Create_MSRCR, 0);

    return "Retinex";
}
