# Workflow recipe database  (task -> model -> node clusters)

- Tasks: 30 | task+model recipes: 176
- Self-contained: every recipe has user_intent + description + node clusters. No human annotation step.

# API / Partner Nodes - Image Edit  (`api_partner_nodes_image_edit`)  -  108 workflow(s), 14 model(s)

## API / Partner Nodes - Image Edit / Generic  (`api_partner_nodes_image_edit__generic`)  -  63 workflow(s)  -  source: mixed
- execution: api (API nodes: BeebleSwitchXImageEdit, BriaEraser, BriaExpandImage, BriaGenFill, BriaImageEditNode, BriaIncreaseResolution, BriaProductHolding, BriaRemoveImageBackground, BriaVirtualTryOn, ByteDanceSeedNode, ByteDanceSeedreamNodeV2, ClaudeNode, GeminiImage2Node, GeminiNanoBanana2, GeminiNanoBanana2V2, GrokImageEditNodeV2, GrokImageNode, HunyuanImageEditApi, LumaImageEditNode2, LumaImageModifyNode, LumaImageNode, LumaImageNode2, LumaReferenceNode, MetaMuseImageEditApi, OpenAIChatConfig, OpenAIChatNode, OpenAIGPTImageNodeV2, OpenRouterImageNode, OpenRouterLLMNode, QuiverImageToSVGNodeV2, RecraftCreateStyleNode, RecraftStyleV3InfiniteStyleLibrary, RecraftTextToImageNode, RecraftV4CreateStyleNode, RecraftV4TextToImageNode, RunwayTextToImageNode, WavespeedImageUpscaleNode)
- when to use: Use to generate an image.
- example request: "build an image workflow"
- description: Claude Fable 5, Anthropic's first public Mythos-class frontier model, powers end-to-end creative workflows in ComfyUI. It takes 1 text prompt and generates 1 structured output including storyboard breakdowns, precise prompts for SD models and ControlNet, and custom node scripts. Ideal for concept artists refining visual narratives, AI filmmakers maintaining style consistency across series, and digital creators automating their entire rendering pipeline. | Convert a raster image into a clean, editable SVG vector using Arrow 2, taking one image and returning one vector output. The model produces cleaner geometry and stronger composition with fewer control points. Ideal for vectorizing sketches, logos, and line art, or preparing scalable brand and product assets. | Converts an image input into an SVG output. | Create static ads for your product in the style of a reference advertisement | Describe a scene or world using GPT-5.6 Terra, a balanced multimodal model that interprets one uploaded image to guide visual storytelling and style consistency. Ideal for worldbuilding, mood board creation, and maintaining cohesive aesthetics across a series of works. | Dress the person in a photo with a garment from a separate product shot using Bria. One person photo plus one to three garments or accessories produces a worn look in which the face, pose and background stay untouched and the garment is refitted to the body. Ideal for fashion catalogs and lookbooks without a photoshoot. | Edit an image with Bria FIBO 2.0 by describing the change in plain language, with an optional mask to confine it to one region. FIBO 2.0 is one model for both editing and generation, trained on structured VGL captions, so it holds everything you did not ask to change, and it returns that structured prompt so the edit can be rebuilt or refined from JSON. | Edit an image with precision using Muse Image 1.0, following natural language instructions to alter specific elements while preserving the rest of the scene. Takes 2 input images and produces 1 edited output image. Ideal for product photography touch-ups, creative style transfers, and iterative visual refinement. | Edit existing images with Uni-1's Image Edit mode, preserving composition and core elements. Supports targeted adjustments like lighting and color using prompts and reference images. | Edit images with MAI Image 2.6 Flash, Microsoft's fast image-to-image model that transforms a single input image into a refined output. Ideal for quick retouching, style adjustments, and rapid concept iteration. | Edit images with MAI Image 2.6, Microsoft's flagship model for image editing, combining two reference images into a single refined output. Ideal for multi-reference compositing, product visual restyling, and text-in-image design work. | Edit images with the HY Image 3.5 Preview model, which deeply understands user intent through optimized text-and-image organization to deliver high-quality results. Takes 1 image input and produces 1 image output. Ideal for portrait editing, commercial posters, and film/TV marketing materials. | Engage with OpenAI's advanced language models for intelligent conversations. | Enhance image quality with advanced text rendering and creative control using Grok Imagine Image Quality. Input images or prompts for improved detail and fidelity. | Expand an image beyond its original borders using Bria's outpainting API, generating new surrounding content that matches the original lighting, perspective, and texture while leaving the source untouched. Choose from preset aspect ratios, custom dimensions, or manual canvas placement, with optional text prompts to guide the expansion direction or an auto-generated prompt when left blank. Ideal for extending compositions for print layouts, creating panoramic versions of existing photos, and adapting artwork to new formats or social media dimensions. | GPT-5.6 Luna streamlines visual pre-production by analyzing a single reference image and generating structured creative text from a user prompt. It takes 1 input image and produces no explicit visual outputs, focusing instead on rapid, standardized copywriting. Ideal for batch scene description expansion, shot list organization, and high-volume creative ideation. | GPT-5.6 Sol is a multimodal model workflow for visual storytelling, taking 1 input image and a user prompt to generate detailed scene descriptions and narrative sequences. Ideal for world-building, multi-shot storyboarding, and maintaining consistent art style across long-form illustration or video projects. | Generate brand-new original images using the Luma UNI-1 node with up to 9 reference images for style, character, color, lighting, and composition. Create unique compositions with detailed prompts, adjustable aspect ratios, and seed control. | Generate detailed image descriptions and creative briefs using Kimi K3, a multimodal model with million-token context windows for unified text and image understanding. Input one image and a text prompt to produce nuanced visual analyses or batch descriptions. Ideal for streamlining multi-shot visual series, generating automated script prompts for node-based workflows, and iteratively refining creative concepts through image feedback. | Generate high-contrast, clickable YouTube thumbnails using Nano Banana Pro. Creates dramatic compositions with exaggerated subjects and bold color strategies for maximum engagement. | Generate high-quality images from text prompts using Runway's AI model. | Generate images by blending style references with precise control using Luma Photon. | Generate images from input images using OpenAI GPT Image 1 API. | Generate images from multiple inputs using OpenAI GPT Image 1 API. | Generate images in the style of one or more reference images using Recraft V4 Styles, capturing texture, composition, and rendering without any training or fine-tuning. Feed in multiple reference images to sharpen or broaden the style match, and receive a single styled output. Ideal for brand-consistent asset creation, matching a client's existing campaign look, or adapting styles from any external image source. | Generate new images based on reference styles and compositions with Runway's AI. | Generate polished visual assets and optimize image-analysis prompts with GPT-6 Astra, an AI model that excels at computer vision tasks while adapting to your preferred style and reference images. Input your source visuals and creative direction to receive refined, immediately usable outputs with accurate detail and concise messaging. Ideal for enhancing prompt engineering workflows, preparing image datasets for analysis, and producing presentation-ready visual content that meets professional standards. | Grok 4.5 is a flagship multimodal model designed for complex engineering and creative tasks, with native visual parsing and high-speed inference. It takes creative briefs and generated images as inputs, producing standardized scene descriptions and iterative visual analysis. Ideal for multi-shot storyboarding, batch image description generation, and feedback-driven creative refinement. | Grok Imagine Image 2.0 performs precise image editing and text rendering using xAI's Aurora engine, letting you modify specific elements, swap backgrounds, or restyle scenes through natural language with crisp short-text output. It takes up to 3 reference images and generates at 1K or 2K resolution across 14 aspect ratios, with strong character consistency across scenes. Ideal for poster and cover design, product and brand visuals, or multi-scene character workflows. | Guide image generation using a combination of images and prompt. | High-fidelity image editing template. Preserve original character pose, scene, lighting and composition. Only modify partial elements such as outfit, hairstyle, facial features or camera angle. Great for sequential frame generation with strong subject consistency. | Input a black and white illustration and generate a colored output. | Input an image or video for analysis. Receive detailed text output describing visual content, reasoning, and structured information extraction. | Input your query or image for analysis. Generate a natural language response powered by Anthropic Claude, supporting both text conversation and image understanding. | Leverage Claude Sonnet 5 for high-value autonomous task execution, bridging the gap to flagship performance at a lower cost. Input any text prompt or script to generate precise image descriptions, automated workflow scripts, shot sequence logs, or node logic corrections. Ideal for batch production tasks, debugging ComfyUI node logic, and maintaining consistent output quality across repetitive generation workflows. | Load an image and enter a prompt to interact with Claude Opus 5, a highly capable and cost-efficient AI model that excels at coding, knowledge work, and problem-solving. Ideal for software engineering tasks, scientific research requiring visual analysis, and everyday productivity with a thoughtful, proactive assistant. | Place a product into the hands of the person in a photo with Bria. One person photo plus one to three product shots is enough for a lifestyle image in which the face, pose and background stay untouched and the product is held naturally. Ideal for e-commerce and social content that would otherwise need a shoot. | Remove unwanted objects or regions from images with Bria's Eraser, which fills the masked area with surrounding context while leaving the rest of the image pixel-perfect. Upload 1 image, paint over the target object with a brush mask, and receive 1 cleaned output alongside a before-and-after comparison. Ideal for cleaning up photobombs, removing watermarks or text, and decluttering product or real-estate photos. | Run Claude Fable 5.1, Anthropic's frontier coding and agentic research model, directly in your workflow to tackle long-running, multi-step problems with state-of-the-art reasoning. This template provides access to the model's full capabilities, from root-cause software debugging and complex code generation to scientific analysis and knowledge work, with adjustable effort levels to balance cost and performance. Ideal for autonomous coding agents, deep technical investigations, and extended research tasks that require sustained, high-quality output. | Template for image rework & partial modification. Keep scene, pose, lighting and composition unchanged, only edit outfit, hairstyle or local details. Fast preview and iterative style adjustment powered by GPT-Image-2.5-Flare. | This workflow pairs a single uploaded image with a text prompt, sending both to Claude Opus 5.5 for multimodal reasoning and response generation. Ideal for image captioning, visual question answering, and prompt-driven image analysis. | Transform clothing photos into professional mall billboard advertisements featuring realistic fashion models. | Transform real building photos into architectural blueprints and then into detailed physical scale models. A complete architectural visualization pipeline from photo to miniature. | Transform uploaded images into realistic poster designs. Change character poses, faces, and clothing while maintaining artistic style and color schemes. Outputs a visually impactful poster with artistic typography. | Upload 1-5 style references and create a style ID to generate consistent, on brand images. | Upload a dieline of your product and assemble into a 3D package. | Upload a graphic design and select your desired aspect ratio. Generates and recomposes the graphic to fit the new aspect ratio. | Upload a photo of a character holding a product and your brands product. Generate an image with the products swapped. | Upload a photo of your character and your product. Generate an image of that character holding the product. | Upload a product image and a cinematic style reference. Generate a sequence of stylized product advertisements with consistent composition and lighting. | Upload a reference image and write a prompt to create campaign-ready posters, product shots, and multilingual signage. The model renders clean text and photorealistic outputs suitable for commercial work. | Upload an AI-generated image to restore details and upscale. The workflow preserves colors and sharpness while minimizing artifacts. Output is a high-quality, enlarged version of your input. | Upload an image and generate 12 stylized variations in under 15 seconds. | Upload an image to automatically remove its background using the Bria API. The workflow processes the request asynchronously and returns the edited image with a transparent background. | Upload an image to edit with the Grok model. Generate a modified version with expressive, cinematic visuals | Upload multiple images to create a consistent style reference. Generate new images using this reference to maintain visual coherence across outputs. | Upload your character image and a reference style/lighting image. Generate an edited output with the reference lighting and style applied to your original image. | Upload your character portrait and outfit reference. Generate a new portrait with the specific outfit applied for consistent wardrobe control. Affordable alternative to Nano Banana Pro and Nano Banana 2. | Upload your character, scene and product. Integrates all images into a fashion editorial style photograph. | Upscale any image to higher resolutions using Bria's dedicated resolution enhancement API, which preserves original content without adding new details or regenerating the image. The workflow accepts a single input image and produces one enhanced, higher-resolution output, supporting JPEG and PNG files in RGB, RGBA, or CMYK color modes up to a maximum area of 8192x8192 pixels. Built-in content moderation scans both the input image and generated output to ensure safe, policy-compliant processing. Ideal for preparing images for print, enlarging photos for digital display, or increasing resolution of assets for professional design work. | Use Bria's Generative Fill API to regenerate a specific region of an image while keeping everything outside the mask pixel-perfect unchanged. Paint a mask over the target area, write a detailed prompt for the replacement content, and compare the before/after side-by-side before exporting. Ideal for product photography corrections, object removal or replacement, and creative compositing where precise control over the edited region matters. | [API] Storyboard generation via gpt-image-2.5-flare. Eight image refs -> Final image. Use this to batch-create consistent panels from a set of visual inputs. | [API] image edit via GPT Image 2. 1 input image -> 1 output image. Pick this for cloud-based editing using OpenAI's API.
- member workflows:
    - api_anthropic_claude
    - api_anthropic_claude_fable5
    - api_anthropic_claude_fable5_1
    - api_anthropic_claude_opus_5
    - api_anthropic_claude_opus_5_5
    - api_anthropic_claude_sonnet5
    - api_beeble_switchx_image_edit
    - api_bria_eraser
    - api_bria_expand_image
    - api_bria_genfill
    - api_bria_image_edit
    - api_bria_product_holding
    - api_bria_virtual_try_on
    - api_bytedance_seed
    - api_from_photo_2_miniature
    - api_grok_image_edit
    - api_grok_imagine_image_2_image_edit
    - api_grok_imagine_image_quality_image_edit
    - api_i2i_imageEdit_OpenAi_GPT2
    - api_luma_photon_i2i
    - api_luma_photon_style_ref
    - api_luma_uni_1_image_create
    - api_luma_uni_1_image_edit
    - api_meta_muse_image_image_edit
    - api_openai_chat
    - api_openai_fashion_billboard_generator
    - api_openai_gpt6_astra
    - api_openai_gpt_image_25_flare_image_edit
    - api_openai_gpt_image_25_sunburst_image_edit
    - api_openai_gpt_image_2_image_edit
    - api_openai_image_1_i2i
    - api_openai_image_1_multi_inputs
    - api_openrouter_gpt5_6_luna
    - api_openrouter_gpt5_6_sol
    - api_openrouter_gpt5_6_terra
    - api_openrouter_grok4_5
    - api_openrouter_kimi_k3
    - api_openrouter_mai_image_26_flash_image_edit
    - api_openrouter_mai_image_26_image_edit
    - api_quiver_arrow2_image_to_svg
    - api_quiver_image_to_svg
    - api_recraft_style_reference
    - api_recraft_v4_style_reference
    - api_runway_reference_to_image
    - api_runway_text_to_image
    - api_tencent_hy_image_3_5_preview_image_edit
    - api_wavespeed_seedvr2_ai_image_fix
    - rain_platform_storyboard
    - template-recraft_create_style
    - template_contact_sheet-step_1.app
    - template_eric_seedance_5_subject_and_outfit_combine
    - template_eric_thumbnail_generator
    - template_sferro21_product_ad.app
    - templates-1_input-multiple_styles_prompt.app
    - templates-assemble_dieline
    - templates-color_illustration
    - templates-product_ad-v2.0
    - templates-subject_holding_product.app
    - templates-subject_product_swap.app
    - templates_doc_workbox_poster_recreator
    - templates_graphic_design_recomposer
    - utility_bria_increase_resolution
    - utility_bria_remove_image_background
- node clusters (required structure):
    - (none resolved)
- optional roles: GrokImageEditNodeV2, SaveImage, LoadImage, LumaReferenceNode, MarkdownNote, ByteDanceSeedreamNodeV2, GeminiImage2Node, OpenAIGPTImageNodeV2, RecraftStyleV3InfiniteStyleLibrary, RegexReplace, ResizeAndPadImage, BatchImagesNode
- unresolved nodes: MarkdownNote, Note, PrimitiveNode

## API / Partner Nodes - Image Edit / Nano-Banana  (`api_partner_nodes_image_edit__nano_banana`)  -  12 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiImage2Node, GeminiImageNode, GeminiInputFiles, GeminiNanoBanana2, GeminiNanoBanana2V2, GeminiNode)
- when to use: Use to edit an existing image using Nano-Banana, Gemini.
- example request: "build an image workflow using Nano-Banana"
- description: Create orthographic multi-view character turnaround sheet from one reference image, keep consistent character features across front, side and back views, for sci-fi mech and armor character design. | Design the text layout for your magazine cover photo, and explore packaging options for it. | Edit an image using Nano Banana 2 Lite, the fastest image generation model from Google's Gemini series, producing a side-by-side comparison of the original and edited result. I | Edit images with natural language and mask-based local edits powered by Nano Banana 2.1, taking one input image and returning one edited result alongside a side-by-side comparison view. Ideal for brand marketing posters, e-commerce product shots, and multi-round retouching where subjects and text must stay consistent. | Local style transfer FOR FULL BODY SHOTS via Nano-Banana Pro (Gemini). 1 video (layout reference) + 7 images (style + hero elements) -> 2 image outputs. Transfers the style reference onto the first video frame while integrating the look of hero element references. | Nano-banana (Gemini-2.5-Flash Image) - image editing with consistency. | Nano-banana Pro (Gemini 3.0 Pro Image) - Studio-quality 4K image generation and editing with enhanced text rendering and character consistency. | Start with an image and an empty prompt, then describe the edit you want. Nano Banana 2 generates a modified image while preserving subject consistency and high visual fidelity. | Upload an AI-generated image to upscale and enhance details. Generate a 4K output with improved sharpness and coherence using Nano Banana Pro. | Upload an image and a text prompt to edit it. Generate a new image that modifies the original based on your instructions, maintaining subject consistency and high visual fidelity. | [API] image edit via Nano-Banana and GeminiNanoBanana2 nodes. 1 image -> 1 image. Select for instant cloud rendering without local VRAM requirements. | [API] style transfer via Nano Banana 2 and Gemini nodes. Images or video paths -> styled images. Transfers a reference style onto your image; pick over plain Gemini edit.
- member workflows:
    - api_google_gemini_image
    - api_google_nano_banana2_image_edit
    - api_google_nano_banana2_image_edit_continuation
    - api_google_nano_banana_create_turnaround_sheet
    - api_i2i_imageEdit_nanoBanana2
    - api_nano_banana_2_1_image_edit
    - api_nano_banana_2_lite_image_edit
    - api_nano_banana_pro
    - bepic_templates__bEpic_styletransfer_NanoBananaPro
    - styletransfer_NanoBananaPro
    - template-multistyle-magazine-cover-nanobananapro
    - utility_nanobanana_pro_ai_image_fix
- node clusters (required structure):
    - (none resolved)
- optional roles: VHS_LoadImagePath, GeminiImage2Node, SaveImage, BatchImagesNode, GeminiInputFiles, LoadImage, MarkdownNote, GeminiImageNode, GeminiNanoBanana2, GeminiNanoBanana2V2, GeminiNode, ImageCompare
- unresolved nodes: ImageRemoveAlpha+, MarkdownNote, Reroute

## API / Partner Nodes - Image Edit / Flux  (`api_partner_nodes_image_edit__flux`)  -  7 workflow(s)  -  source: official
- execution: api (API nodes: FluxEraseNode, FluxKontextMaxImageNode, FluxKontextProImageNode, FluxProExpandNode, FluxProUltraImageNode, FluxVTONode)
- when to use: Use to generate an image using Flux.
- example request: "build an image workflow using Flux"
- description: Edit images with Flux.1 Kontext max image. | Edit images with Flux.1 Kontext pro image. | Generate images with excellent prompt following and visual quality using FLUX.1 Pro. | Input multiple images and edit them with Flux.1 Kontext. | Upload a person photo and a garment image. Generate a virtual try-on result with the garment applied to the person while preserving face and pose. | Upload an image and paint a mask over the object to remove. The workflow erases the masked area and reconstructs the background. | Upload an image to expand its canvas using Flux.1. Generate an outpainted image with extended content.
- member workflows:
    - api_bfl_flux1_expand_image
    - api_bfl_flux_1_kontext_max_image
    - api_bfl_flux_1_kontext_multiple_images_input
    - api_bfl_flux_1_kontext_pro_image
    - api_bfl_flux_pro_t2i
    - api_flux_erase_image
    - api_flux_vto
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
- optional roles: MarkdownNote, ImageStitch, FluxEraseNode, FluxKontextMaxImageNode, FluxKontextProImageNode, FluxProExpandNode, FluxProUltraImageNode, FluxVTONode, ImageCompare, PreviewImage
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Seedream  (`api_partner_nodes_image_edit__seedream`)  -  6 workflow(s)  -  source: mixed
- execution: api (API nodes: ByteDanceSeedreamLayerSeparationNodeV2, ByteDanceSeedreamNode, ByteDanceSeedreamNodeV2, ByteDanceSeedreamNodeV3)
- when to use: Use to edit an existing image using Seedream.
- example request: "build an image workflow using Seedream"
- description: Edit and generate images with Seedream 5.0 Flash, transforming a single input image into one refined output with cinematic lighting, accurate text rendering, and resolutions up to 2K. | Edit images with precision using Seedream 5.0 Pro, targeting specific regions while preserving lighting, depth, and texture. The workflow takes 1 input image and produces 1 output image. Ideal for product photography edits, portrait consistency adjustments, and creating structured layouts with multilingual text. | Multi-modal AI model for text-to-image and image editing. Generate 2K images in under 2 seconds with natural language control. | Split any image into independent, editable layers - subject, product, background, and text - with a single text instruction using Seedream 5.0 Pro. A single poster can yield 10+ layers, with design-grade text separation supporting multilingual headlines and CJK fonts. Ideal for brand design, e-commerce asset creation, and layout iteration. | Upload an image of your product and background. Composite them and seamlessly relight and fuse together using Seedream 4.5. | [API] Image edit via Seedream. 2 inputs -> 1 output. Use this for cloud-based generation when local hardware is limited.
- member workflows:
    - api_bytedance_seedream4
    - api_bytedance_seedream_5_0_flash_image_edit
    - api_bytedance_seedream_5_0_layer_separation
    - api_bytedance_seedream_5_0_lite_image_edit
    - api_bytedance_seedream_5_0_pro_image_edit
    - templates-product_scene_relight
- node clusters (required structure):
    - inputs: LoadImage
- optional roles: MarkdownNote, BatchImagesNode, BatchMasksNode, ByteDanceSeedreamLayerSeparationNodeV2, ByteDanceSeedreamNode, ByteDanceSeedreamNodeV2, ByteDanceSeedreamNodeV3, ImageCompare, ImageCompositor, JoinImageWithAlpha, Note, Painter
- unresolved nodes: MarkdownNote, Note

## API / Partner Nodes - Image Edit / Ideogram  (`api_partner_nodes_image_edit__ideogram`)  -  4 workflow(s)  -  source: official
- execution: api (API nodes: IdeogramEditApi, IdeogramPreciseEditApi, IdeogramVectorGenerateApi, IdeogramVectorImageToSVGApi)
- when to use: Use to edit an existing image using Ideogram.
- example request: "build an image workflow using Ideogram"
- description: Generate a fully editable SVG from a text prompt with Ideogram Vector 1, optionally guided by up to five reference images, and save it as one vector file. Ideal for logos, posters, flat illustration, patterns and line art that stays crisp at any size and opens directly in a vector editor. | Generate a precisely edited image from a text prompt and up to five input images using Ideogram 4.5, where image 1 is the image to edit and images 2-5 are optional references. Only what the prompt asks for changes, so untouched pixels stay identical and the output keeps the size of image 1, with a choice of quality tiers. Ideal for product recolours and material swaps, replacing one element of a photo without disturbing its lighting, background, or framing, and touch-ups that must leave the rest of the image alone. | Generate edited or combined images from a text prompt and up to five input images using Ideogram 4.5, where image 1 is the image to edit and images 2-5 are references for design, material, or colour. Canvas size, quality level, and seed are adjustable, and one image is output. Ideal for product and outfit swaps, applying a reference design or palette to an existing photo, and recomposing a shot into a different aspect ratio. | Trace a bitmap into a fully editable SVG with Ideogram Vector 1 and save it as one vector file. Ideal for turning logos, flat illustration, patterns and line art into clean paths that stay crisp at any size, with a detail mode that either keeps large flat color blocks or finer color regions. Photographs also work, but convert into many small paths, so clean sources with large areas of flat color and transparent backgrounds vectorize best.
- member workflows:
    - api_ideogram_v4_5_image_edit
    - api_ideogram_v4_5_precise_image_edit
    - api_ideogram_vector_1_image_to_svg
    - api_ideogram_vector_1_text_to_svg
- node clusters (required structure):
    - inputs: LoadImage
    - other operations: MarkdownNote
- optional roles: BuildJsonPromptIdeogram, CreateBoundingBoxes, IdeogramEditApi, IdeogramPreciseEditApi, IdeogramVectorGenerateApi, IdeogramVectorImageToSVGApi, ImageCompare, SaveImageAdvanced, SaveSVGNode
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Flux 2  (`api_partner_nodes_image_edit__flux_2`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Flux2ImageNode)
- when to use: Use to generate an image using Flux 2.
- example request: "build an image workflow using Flux 2"
- description: Generate up to 4MP photorealistic images with multi-reference consistency and professional text rendering. | Replace objects in images with unmatched quality using FLUX.2 [max]. Perfect for product photography, furniture swaps, and maintaining scene consistency with highest editing precision.
- member workflows:
    - api_bfl_flux2_max_sofa_swap
    - api_flux2
- node clusters (required structure):
    - inputs: LoadImage (x3)
    - output: SaveImage
    - other operations: Flux2ImageNode, MarkdownNote
- paired/multiple required: LoadImage x3
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Flux 3  (`api_partner_nodes_image_edit__flux_3`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Flux3ImageNode)
- when to use: Use to edit an existing image using Flux 3.
- example request: "build an image workflow using Flux 3"
- description: Edit or combine up to ten reference images from a text instruction with FLUX 3 Image, referring to them in the prompt as image 1, image 2, and so on. One image is output at any of sixteen aspect ratios and up to 4K, with web and image search grounding on by default. Ideal for character-consistent edits, product and object placement, and multi-reference scene compositing. | Place objects and text where you want them with FLUX 3 Image by feeding boxes from the Create Bounding Boxes node into the prompt, so each box carries its own description and copy into the render. Boxes are relative to the canvas, so set the output to the same aspect ratio. One image is output, up to 4K. Ideal for poster and layout composition, placing headlines and captions, and controlling where subjects sit in the frame.

- member workflows:
    - api_bfl_flux3_image_bboxes
    - api_bfl_flux3_image_edit
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImageAdvanced
    - other operations: Flux3ImageNode, ImageCompare, MarkdownNote
- optional roles: CreateBoundingBoxes
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Flux Krea  (`api_partner_nodes_image_edit__flux_krea`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Krea2ImageNode, Krea2StyleReferenceNode, Krea3ImageNode)
- when to use: Use to generate an image using Flux Krea.
- example request: "build an image workflow using Flux Krea"
- description: Perform targeted regional edits on an image with Krea 3, changing only the areas you specify while the rest of the picture stays untouched. One image goes in and one edited image comes out. Ideal for e-commerce product restaging, fashion and portrait relighting, and architectural sketch-to-render work. | Upload a style reference image and input a text prompt. Generate an image matching the prompt while applying the aesthetic style from your reference.
- member workflows:
    - api_krea2_style_reference
    - api_krea3_image_edit
- node clusters (required structure):
    - inputs: LoadImage
    - other operations: MarkdownNote
- optional roles: Krea2StyleReferenceNode, Krea2ImageNode, Krea3ImageNode, SaveImage, SaveImageAdvanced
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Gemini  (`api_partner_nodes_image_edit__gemini`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: GeminiNodeV3, OpenRouterLLMNode)
- when to use: Use to produce an image using Gemini.
- example request: "build an image workflow using Gemini"
- description: Experience Google's multimodal AI with Gemini's reasoning capabilities. | Select a model from OpenRouter's curated list (Claude, GPT, Gemini, etc.). Generate a text response with optional media uploads for vision-capable models.
- member workflows:
    - api_google_gemini
    - api_openrouter_llm
- node clusters (required structure):
    - inputs: LoadImage
- optional roles: GeminiNodeV3, MarkdownNote, OpenRouterLLMNode, SaveText
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Kling  (`api_partner_nodes_image_edit__kling`)  -  2 workflow(s)  -  source: mixed
- execution: api (API nodes: KlingOmniProImageNode, KlingTryOnNode)
- when to use: Use to generate an image using Kling.
- example request: "build an image workflow using Kling"
- description: Dress a virtual model in any garment by combining a clothing image with a person photo, generating a realistic try-on result alongside a side-by-side comparison view. It accepts flat-lay, mannequin, or on-model clothing shots while preserving the model's face and pose. Ideal for fashion e-commerce previews, styling mockups, and virtual fitting experiences.
- member workflows:
    - api_kling_o3_image
    - api_kling_vto
- node clusters (required structure):
    - inputs: LoadImage (x2)
- paired/multiple required: LoadImage x2
- optional roles: ImageBatchMulti, ImageCompare, KlingOmniProImageNode, KlingTryOnNode, MarkdownNote, SaveImage, SaveImageAdvanced
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image Edit / Magnific  (`api_partner_nodes_image_edit__magnific`)  -  2 workflow(s)  -  source: mixed
- execution: api (API nodes: MagnificImageRelightNode, MagnificImageStyleTransferNode)
- when to use: Use to transfer a style onto an image using Magnific.
- example request: "build an image workflow using Magnific"
- description: API image relighting via Magnific. 1 source image + 1 lighting reference image -> 1 relit image output. Applies the lighting conditions from the reference onto the source image. | Upload a target image and a style reference image. Generate a new image that merges the artistic style of the reference with the structure of your target.
- member workflows:
    - api_magnific_image_relight
    - api_magnific_image_style_transfer
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - output: SaveImage
- paired/multiple required: LoadImage x2
- optional roles: MagnificImageRelightNode, MagnificImageStyleTransferNode

## API / Partner Nodes - Image Edit / Z-Image  (`api_partner_nodes_image_edit__z_image`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: GrokImageNode)
- when to use: Use to generate an image using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Upload a reference image, uses Grok to caption and generate variations with a fast 2x upscale for added details. | Upload an image that you want a variation of. The workflow captions the input image, generates with Grok Image, and upscales with Z-Image to add realistic textures.
- member workflows:
    - template_rob_realistic_2k_images_quick_variations
    - templates_rob_realistic_2k_images_quick_variations.app
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, UNETLoader, UpscaleModelLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage (x2)
    - other operations: AILab_QwenVL, GrokImageNode, ImageCompare, ImageRemoveAlpha+, ImageScaleBy, ImageScaleToTotalPixels, ImageUpscaleWithModel, ModelSamplingAuraFlow, Note, PreviewAny, StringConcatenate (x2)
- paired/multiple required: CLIPTextEncode x2, SaveImage x2
- unresolved nodes: AILab_QwenVL, ImageRemoveAlpha+, Note

## API / Partner Nodes - Image Edit / Flux 2 Klein  (`api_partner_nodes_image_edit__flux_2_klein`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: ByteDanceSeedreamNode, GeminiImage2Node, GrokImageEditNode, OpenAIGPTImage1)
- when to use: Use to edit an existing image using Flux 2 Klein, Qwen Image.
- example request: "build an image workflow using Flux 2 Klein"
- description: upload one input image and select multiple editing models. generate side-by-side outputs to directly compare model effects and quality.
- member workflows:
    - templates-all_in_one-image_edit_models
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader (x3), UNETLoader (x3), VAELoader (x3)
    - conditioning: BasicGuider, CFGGuider, CLIPTextEncode (x3), FluxGuidance, TextEncodeQwenImageEditPlus (x2)
    - latent / canvas: VAEEncode (x3)
    - sampling: KSampler, KSamplerSelect (x2), SamplerCustomAdvanced (x2)
    - decoding: VAEDecode (x3)
    - output: SaveImage (x7)
    - other operations: ByteDanceSeedreamNode, CFGNorm, EmptyFlux2LatentImage (x2), Flux2Scheduler (x2), FluxKontextImageScale, FluxKontextMultiReferenceLatentMethod (x2), GeminiImage2Node, GetImageSize (x2), GrokImageEditNode, ImageScaleToTotalPixels (x2), MarkdownNote (x2), ModelSamplingAuraFlow, Note, OpenAIGPTImage1, PrimitiveStringMultiline, RandomNoise (x2), ReferenceLatent (x3), Reroute (x2)
- paired/multiple required: SaveImage x7, CLIPLoader x3, CLIPTextEncode x3, ReferenceLatent x3, UNETLoader x3, VAEDecode x3, VAEEncode x3, VAELoader x3, EmptyFlux2LatentImage x2, Flux2Scheduler x2, FluxKontextMultiReferenceLatentMethod x2, ImageScaleToTotalPixels x2, KSamplerSelect x2, MarkdownNote x2, RandomNoise x2, SamplerCustomAdvanced x2, TextEncodeQwenImageEditPlus x2
- unresolved nodes: MarkdownNote, Note, Reroute

## API / Partner Nodes - Image Edit / Qwen Image  (`api_partner_nodes_image_edit__qwen_image`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: QwenImageEditApi)
- when to use: Use to edit an existing image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Qwen Image 3.0 Pro performs advanced image editing with rich multi-image input, generating complex layouts like newspapers, storyboards, menus, and exam papers in a single pass while preserving fine details down to 10px text, micro-expressions, and individual hairs. Supports native rendering across 12 languages with realistic simulation of web pages, games, and live streams. Ideal for professional content production, interface mockups, and high-fidelity commercial imagery requiring authentic detail.
- member workflows:
    - api_qwen3_image_edit
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: QwenImageEditApi


# Text to Image  (`text_to_image`)  -  86 workflow(s), 16 model(s)

## Text to Image / Generic  (`text_to_image__generic`)  -  32 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image from a text prompt.
- example request: "build an image workflow"
- description: A lightweight 2B model that generates images from English and Russian prompts with high visual quality. | Chroma1-Radiance works directly with image pixels instead of compressed latents, delivering higher quality images with reduced artifacts and distortion. | Demonstrate how to convert between various built-in data types within a workflow. Input any data value and output its converted form. | Enhance SDXL images using refiner models. | Generate detailed anime-style images with NewBie Exp0.1's Next-DiT architecture. Supports XML structured prompts for better multi-character scenes and attribute control. | Generate high-quality images from text prompts using OmniGen2's unified 7B multimodal model with dual-path architecture. | Generate high-quality images from text prompts using the Boogu Turbo model, a distilled 10B-parameter text-to-image pipeline that produces photorealistic outputs in just 4 steps. The workflow accepts one text prompt and outputs a single 1024x1024 image. Ideal for rapid prototyping of photorealistic concepts, generating marketing visuals, and creating bilingual text-rendered images with strong prompt adherence. | Generate high-quality images from text prompts using the SDXL 1.0 model, offering a straightforward text-to-image generation workflow. Ideal for quick concept visualization, creative brainstorming, and rapid prototyping of visual ideas. | Generate high-quality images from text prompts using the compact 4B-parameter Mage-Flow Turbo model, producing a single image output without any file inputs. This workflow supports native-resolution generation from 512 to 2048 pixels per side at any aspect ratio, including extreme 4:1 dimensions. Ideal for rapid concept visualization, creative asset generation, and low-latency image prototyping on modest hardware. | Generate high-quality images using SDXL. | Generate high-resolution images from text prompts using Mage-Flow, a compact 4B-parameter diffusion model that outputs a single image at native resolutions from 512 to 2048 pixels per side. Ideal for rapid concept visualization, creative asset generation, and research in controlled image synthesis under limited compute budgets. | Generate images in a single step using SDXL Turbo. | Generate images quickly with HiDream I1 Fast - Lightweight version with 16 inference steps, ideal for rapid previews on lower-end hardware. | Generate images using SD 3.5. | Generate images with HiDream I1 Dev - Balanced version with 28 inference steps, suitable for medium-range hardware. | Generate images with HiDream I1 Full - Complete version with 50 inference steps for highest quality output. | Generate text-heavy design compositions from text prompts using Ming Image 0.1 Design, a 6B model tuned for UI screens, infographics, and posters. It takes no file inputs and produces one image, supporting transparent backgrounds for easy layering. Ideal for interface mockups, marketing posters, and data-rich infographic drafts. | Input a text prompt and optional negative prompt. Generate a 1024px image using PixelDiT's VAE-free pixel diffusion transformer. | Input a text prompt and select resolution and aspect ratio. Generate a high-quality image using the efficient Lens text-to-image model. | Input a text prompt and select resolution, aspect ratio, and inference steps. Generate a high-quality image using the Lens text-to-image model. | Input a text prompt to generate an image. This workflow uses the Capybara unified model for text-to-image synthesis. | Ovis-Image is a 7B text-to-image model specifically optimized for high-quality text rendering in generated images. Designed to operate efficiently under computational constraints, it excels at accurately generating images containing text content. | [API] Intrinsic video decomposition via UniVidX nodes. input video -> intrinsic and alpha channels. Use this to isolate static geometry and material layers from animated backgrounds. | [API] text-to-intrinsic via UniVidX. text -> intrinsic maps. Generates intrinsic scene layers and alpha masks from text prompts for compositing workflows. | [Local] Face detailing via DetailerForEach nodes. image -> 3 images. Apply five distinct prompts to tiled regions for varied results. | [Local] OCIO color convert. 1 EXR (or PNG) in -> 1 PNG out. Loads via bepic_imageLoad (OIIO), applies bepic_colorTransform input ACES - ACEScg to output Output - sRGB with clamp on, saves 16-bit PNG via bEpic_imageSave (OIIO). For batches of non-contiguous frames, run one job per file and patch image_path + first_frame; keep auto_version false so saves go straight into the target folder. | [Local] Video decomposition via UniVidX. input video -> decomposed layers. Isolates surface material and transparency tracks for compositing. | [Local] face detailing via FaceDetailerPipe and SAM loaders. image -> refined image. Automatically detects and refines faces for batch processing, ideal when manual mask creation is too slow. | [Local] material estimation via ChordMaterialEstimation. single image -> multiple material maps. Estimates PBR properties from textures for 3D rendering. | [Local] text-to-image with face detailing via FaceDetailer. text -> image. Use this for automatic facial detail enhancement with iterative upscaling powered by SAM detection models. | [Local] video background removal via AILab_VRMBG. Video in -> images with background removed; pick for VRMBG-3.0 video matting. | [Local] video decomposition via UniVidX. text -> composite, alpha, foreground, background videos. Picks this text-input form for basic matting; use the RGB-fed pipeline for sharper alpha masks.
- member workflows:
    - ComfyUI-Chord__chord_sdxl_t2i_image_to_material
    - ComfyUI-Impact-Pack__1-FaceDetailer
    - ComfyUI-Impact-Pack__5-prompt-per-tile
    - ComfyUI-Impact-Pack__6-DetailerWildcard
    - ComfyUI-RMBG__VRMBG
    - Image_capybara_v0_1_text_to_image
    - UniVidX_ComfyUI__t2RAIN_basic_api
    - UniVidX_ComfyUI__t2RAIN_tiny_api
    - UniVidX_ComfyUI__t2RPFB_basic
    - UniVidX_ComfyUI__t2RPFB_basic_api
    - acescg_to_srgb
    - basic_datatype_conversion
    - hidream_i1_dev
    - hidream_i1_fast
    - hidream_i1_full
    - image_boogu_image_0_1_turbo_t2i
    - image_chroma1_radiance_text_to_image
    - image_kandinsky5_t2i
    - image_lens_t2i
    - image_lens_turbo_t2i
    - image_mage_flow_t2i_int8
    - image_mage_flow_turbo_t2i_int8
    - image_ming_image_01_design_t2i
    - image_newbieimage_exp0_1-t2i
    - image_omnigen2_t2i
    - image_ovis_text_to_image
    - image_pixeldit_t2i
    - image_sdxl_simple
    - sd3.5_simple_example
    - sdxl_refiner_prompt_example
    - sdxl_simple_example
    - sdxlturbo_example
- node clusters (required structure):
    - (none resolved)
- optional roles: Note, SaveImage, PreviewImage, CLIPTextEncode, MarkdownNote, CLIPLoader, CheckpointLoaderSimple, KSamplerAdvanced, StringListToString, WD14Tagger|pysssss, AILab_Preview, AILab_VRMBG
- unresolved nodes: MakeCircularVAE, MarkdownNote, Note, PrimitiveNode, Reroute, SeamlessTile, WD14Tagger|pysssss, workflow>MAKE_BASIC_PIPE

## Text to Image / Qwen Image  (`text_to_image__qwen_image`)  -  13 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Qwen Image, Z-Image.
- example request: "build an image workflow using Qwen Image"
- description: A t2v workflows for a style LoRA built on Qwen-Image. Produces images spanning cute anime character design, European bande dessinee, indie risograph prints, storybook watercolor, retro shoujo manga, and graphic novel illustration. | Apply a Qwen Image LoRA to transform any subject into a sour gummy candy-style character. Enter an animal name in the prompt to generate a stylized, edible-looking creature. | Generate 360-degree equirectangular projection images from text descriptions using a rank 128 LoRA on the 20B MMDiT model. | Generate images from text prompts with Qwen Image 2.1 Turbo, the 8-step accelerated checkpoint, producing a single 1024×1024 image in 8 sampling steps. Same model family with native 2K output, professional typography, and alpha-channel support, plus an optional Qwen 3.5 prompt enhancer. Ideal for poster and layout design, transparent asset creation, and rapid concept visualization. | Generate images from text prompts with Qwen Image 2.1, the latest open-weight Qwen-Image model, producing a single 1024×1024 image over 25 sampling steps. Generation and instruction-based editing share one model with native 2K output, professional typography, and alpha-channel support. Ideal for poster and layout design, transparent asset creation, and rapid concept visualization. | Generate images with exceptional multilingual text rendering and editing capabilities using Qwen-Image's 20B MMDiT model. | Generates images from text prompts using Qwen-Image, Alibaba's 20B MMDiT model with excellent multilingual text rendering. | Generates images from text prompts using Qwen-Image-2512, with enhanced human realism and finer natural detail over the base version. | Generates images from text prompts using Z-Image base weights with Qwen3 text encoder and bundled VAE. | Generates images from text prompts using Z-Image-Turbo defaults with Qwen3 text encoder and VAE. | Input a text prompt to generate detailed, reasoned responses using the Qwen3-4B-Thinking model. | Qwen-Image 2512 Image generation with 2-steps Turbo LoRA. Input your text prompt to quickly produce a result, though some quality is sacrificed for speed. | Text-to-image model with enhanced human realism, finer natural details for landscapes and animal fur, and improved text rendering with accurate layout and multimodal composition.
- member workflows:
    - image_qwen_Image_2512
    - image_qwen_image
    - image_qwen_image_2512_with_2steps_lora
    - image_qwen_image_2_1_t2i
    - image_qwen_image_2_1_turbo_t2i
    - llm_qwen3_text_gen
    - template_qwen_Image_2512_360_lora
    - template_qwen_image_illustration_lora
    - template_sugar_coated_gummy_style_qwen
    - text_to_image
    - text_to_image_qwen_image
    - text_to_image_qwen_image_2512
    - text_to_image_z_image_base
- node clusters (required structure):
    - model loading: CLIPLoader
- optional roles: MarkdownNote, CLIPTextEncode, LoraLoaderModelOnly, CFGGuider, ConditioningZeroOut, EmptyLatentImage, EmptySD3LatentImage, KSampler, KSamplerSelect, ManualSigmas, ModelSamplingAuraFlow, Note
- unresolved nodes: MarkdownNote, Note

## Text to Image / Flux  (`text_to_image__flux`)  -  7 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image from a text prompt using Flux.
- example request: "build an image workflow using Flux"
- description: Chroma - enhanced Flux model with improved image quality and better prompt understanding for stunning text-to-image generation. | Generate high-quality images with Flux Dev full version. Requires larger VRAM and multiple model files, but provides the best prompt following capability and image quality. | Generate images quickly with Flux.1 Schnell full version. Uses Apache2.0 license, requires only 4 steps to generate images while maintaining good image quality. | Generate images using Flux.1 Dev fp8 quantized version. Suitable for devices with limited VRAM, requires only one model file, but image quality is slightly lower than the full version. | Generates images from prompts using FLUX.1 [dev]: a 12B rectified-flow MMDiT with dual CLIP plus T5-XXL text encoders and guidance-distilled sampling for sharp prompt following versus classic DDPM diffusion. | Quickly generate images with Flux.1 Schnell fp8 quantized version. Ideal for low-end hardware, requires only 4 steps to generate images. | [Local] material estimation via ChordMaterialEstimation. image -> material maps. Use it to convert AI textures into PBR-ready asset sheets.
- member workflows:
    - ComfyUI-Chord__chord_flux_t2i_image_to_material
    - flux_dev_checkpoint_example
    - flux_dev_full_text_to_image
    - flux_schnell
    - flux_schnell_full_text_to_image
    - image_chroma_text_to_image
    - text_to_image_flux_1_dev
- node clusters (required structure):
    - latent / canvas: EmptySD3LatentImage
    - decoding: VAEDecode
- optional roles: SaveImage, CLIPTextEncode, Note, BasicScheduler, CFGGuider, CLIPLoader, CLIPTextEncodeFlux, CheckpointLoaderSimple, ChordLoadModel, ChordMaterialEstimation, ChordNormalToHeight, ConditioningZeroOut
- unresolved nodes: MarkdownNote, Note

## Text to Image / Z-Image  (`text_to_image__z_image`)  -  7 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Foundation for creative freedom. Diverse aesthetics with exceptional photorealistic quality; ideal for fine-tuning; responsive to negative prompts; high generation diversity. | Generate high-quality, diverse images from text prompts using the Z-Image Int8 model, which leverages an undistilled transformer for precise prompt adherence and broad stylistic coverage. The workflow takes no file inputs and produces 1 image output through a standard text-to-image pipeline. Ideal for professional creative projects, LoRA training and model development, and scenarios requiring robust negative control and high compositional diversity. | Generate photorealistic images using the Z-Image-Turbo int8 model, a distilled 6B-parameter variant optimized for sub-second inference through efficient 8-step sampling. The workflow takes a text prompt as input and outputs a single high-quality image. Ideal for rapid creative iteration, realistic concept art, and production-ready visual prototyping. | Generates images from text prompts using Z-Image-Turbo, Alibaba's distilled 6B DiT model. | Guide on how Switch Nodes work, simple Text Switch and LoRA Switch | [Local] PBR material extraction via Chord and Z-Image Turbo. image -> PBR maps. Automatically derives full physically based rendering maps from any source texture. | [Local] text-to-image via Z-Image-Turbo. 1 text input -> 1 image output. High-speed image generation from text prompts.
- member workflows:
    - ComfyUI-Chord__chord_zimage_turbo_t2i_image_to_material
    - basic_switch_node
    - image_z_image
    - image_z_image_int8
    - image_z_image_turbo
    - image_z_image_turbo_int8
    - text_to_image_z_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ModelSamplingAuraFlow
- optional roles: SaveImage, MarkdownNote, ChordLoadModel, ChordMaterialEstimation, ChordNormalToHeight, ConditioningZeroOut, LoraLoaderModelOnly, ResizeAndPadImage, ResolutionSelector, SaveImageAdvanced
- unresolved nodes: MarkdownNote

## Text to Image / Anima  (`text_to_image__anima`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Anima.
- example request: "build an image workflow using Anima"
- description: Input a text prompt describing an anime or artistic illustration. Generate a non-photorealistic image focused on anime concepts, characters, or styles. | Input a text prompt to generate an anime-style image using the Anima model. Configure settings like steps and CFG scale to control the output. The workflow produces a non-photorealistic illustration based on your description. | This subgraph converts text prompts into non-photorealistic illustrations using a 2-billion-parameter model optimized for anime and artistic styles. It is ideal for generating concept art, character designs, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects. | This subgraph generates non-photorealistic illustrations from text prompts using a 2-billion-parameter model optimized for anime concepts, characters, and styles. It is ideal for creating artistic images, concept art, or stylized illustrations where photorealism is not required. The model excels with anime and artistic content but performs poorly on realistic subjects.
- member workflows:
    - image_anima_base_v1
    - image_anima_preview
    - text_to_image_anima
    - text_to_image_anima_base_1_0
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentImage
    - sampling: KSampler
    - decoding: VAEDecode
- paired/multiple required: CLIPTextEncode x2
- optional roles: LoraLoaderModelOnly, MarkdownNote, ResolutionSelector, SaveImage
- unresolved nodes: MarkdownNote

## Text to Image / ERNIE  (`text_to_image__ernie`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using ERNIE.
- example request: "build an image workflow using ERNIE"
- description: Faster ERNIE Image Turbo variant (~8B DiT, distilled for fewer sampling steps): same strengths in Chinese/English on-image text and layout-heavy graphics as the base ERNIE Image lineup, with bundled encoders and VAE. | Generate images from text prompts using the ERNIE-Image model. Input a text description to produce detailed, structured visuals with a broad stylistic range. | Generate images from text prompts using the ERNIE-Image turbo model. Input a text description and receive a high-quality image with precise text rendering and structured layouts. | Generates images from text prompts using Baidu's open ERNIE Image (~8B DiT): bilingual in-image typography and layouts (posters, infographics, multi-panel compositions) alongside general scenes, with bundled encoders and VAE.
- member workflows:
    - image_ernie_image
    - image_ernie_image_turbo
    - text_to_image_ernie_image
    - text_to_image_ernie_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader (x2), UNETLoader, VAELoader
    - conditioning: CLIPTextEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ComfySwitchNode, EmptyFlux2LatentImage, PreviewAny (x3), PrimitiveBoolean, PrimitiveStringMultiline, StringReplace (x3), TextGenerate
- paired/multiple required: CLIPLoader x2
- optional roles: MarkdownNote, ConditioningZeroOut, SaveImage
- unresolved nodes: MarkdownNote

## Text to Image / Flux Krea  (`text_to_image__flux_krea`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux Krea.
- example request: "build an image workflow using Flux Krea"
- description: A fine-tuned FLUX model pushing photorealism to the max | FLUX.1 Krea [dev] (Black Forest Labs × Krea): open-weight 12B rectified-flow text-to-image drop-in alongside FLUX.1 [dev], tuned away from overcooked saturation toward more natural diversity in people, realism, and style while keeping ecosystem compatibility. | Generate images from text prompts using Krea 2, a foundation model built for aesthetic quality and creative control. It focuses on rendering expressive and artistic styles. Ideal for concept art exploration, visual brainstorming, and creating stylized imagery for design projects. | Generate images with the Krea-2 Turbo distilled text-to-image model using the high-performance Int8 Convrot format, which delivers better quality than FP8 while generally running faster across most GPUs. This workflow accepts no file inputs and produces a single image output, with optional style LoRA support for creative customization. Ideal for rapid concept art iteration, batch asset generation for games or marketing, and exploring the speed-quality sweet spot of the Convrot Int8 quantization.
- member workflows:
    - flux1_krea_dev
    - image_krea2_turbo_t2i
    - image_krea2_turbo_t2i_int8
    - text_to_image_flux_1_krea_dev
- node clusters (required structure):
    - model loading: UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut
    - sampling: KSampler
    - decoding: VAEDecode
- optional roles: MarkdownNote, CLIPLoader, DualCLIPLoader, EmptyLatentImage, EmptySD3LatentImage, LoraLoaderModelOnly, ResolutionSelector, SaveImage, TextGenerate
- unresolved nodes: MarkdownNote

## Text to Image / Ideogram  (`text_to_image__ideogram`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Ideogram.
- example request: "build an image workflow using Ideogram"
- description: Generate images from text prompts using the Ideogram v4 model, a state-of-the-art open-weight text-to-image system with exceptional text rendering and layout control. No file inputs are required, producing 1 image output. Ideal for creating marketing visuals, generating design mockups with precise typography, and producing high-resolution artwork with controlled compositions. | Input a text prompt or structured JSON description. Generate an image with precise layout, color, and style control using Ideogram 4.0. | This subgraph generates images using Ideogram v4, accepting plain text or structured JSON prompts for precise layout and style control. It suits detailed illustrations, concept art, or marketing visuals needing predictable composition and color palettes. The model uses flow-matching with asymmetric guidance, so no negative prompt is needed, but JSON prompts yield the best results.
- member workflows:
    - image_ideogram4_t2i
    - image_ideogram4_t2i_int8
    - text_to_image_ideogram_v4
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, DualModelGuider
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: CFGOverride, ComfyMathExpression (x2), ComfyNumberConvert (x3), CustomCombo, EmptyFlux2LatentImage, Ideogram4Scheduler, JsonExtractString (x4), PrimitiveInt (x2), RandomNoise, StringReplace
- paired/multiple required: UNETLoader x2
- optional roles: MarkdownNote, ResolutionSelector, SaveImage
- unresolved nodes: MarkdownNote

## Text to Image / Flux 2  (`text_to_image__flux_2`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux 2.
- example request: "build an image workflow using Flux 2"
- description: Generates images from prompts using FLUX.2 [dev]: a newer 32B rectified-flow stack with distilled guidance plus a stronger long-context multimodal encoder for complex scenes, sharper typography/UI text, anatomy, lighting, and high-resolution latent decoding. | Text-to-image with enhanced lighting, materials, and realistic details.
- member workflows:
    - image_flux2_text_to_image
    - text_to_image_flux_2_dev
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, CLIPTextEncode, FluxGuidance
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x2), EmptyFlux2LatentImage, Flux2Scheduler, PrimitiveBoolean, PrimitiveInt (x2), RandomNoise
- optional roles: MarkdownNote, SaveImage
- unresolved nodes: MarkdownNote

## Text to Image / Flux 2 Klein  (`text_to_image__flux_2_klein`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image from a text prompt using Flux 2 Klein.
- example request: "build an image workflow using Flux 2 Klein"
- description: BFL distilled model. Outstanding quality at sub-second speed. Great for real-time generation while retaining quality. Marketing launch will focus on this model.
- member workflows:
    - image_flux2_klein_text_to_image
    - image_flux2_text_to_image_9b
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CFGGuider, CLIPTextEncode (x2)
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: EmptyFlux2LatentImage, Flux2Scheduler, PrimitiveInt (x2), RandomNoise
- paired/multiple required: CLIPTextEncode x2
- optional roles: MarkdownNote
- unresolved nodes: MarkdownNote

## Text to Image / Flux 3  (`text_to_image__flux_3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Flux 3.
- example request: "build an image workflow using Flux 3"
- description: Generate images from text prompts using FLUX 3 Image Klein, producing a single saved image with no file inputs required. Ideal for concept art, marketing visuals, and rapid prototyping of scene ideas from written descriptions. | Generate images from text prompts using the distilled FLUX 3 Image Klein model, which runs efficiently in 15 sampling steps with no classifier-free guidance. Just describe what you want and receive one finished image. Ideal for rapid concept art, marketing visuals, and everyday creative ideation where speed matters.
- member workflows:
    - image_flux3_klein_distilled_t2i
    - image_flux3_klein_t2i
- node clusters (required structure):
    - model loading: CLIPLoader (x2), UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: ComfySwitchNode, CreateBoundingBoxes, EmptyFlux2LatentImage, Flux3LayoutPrompt, MarkdownNote (x2), PreviewAny (x2), PrimitiveStringMultiline (x2), ResolutionSelector, TextGenerate
- paired/multiple required: CLIPLoader x2, CLIPTextEncode x2, MarkdownNote x2
- unresolved nodes: MarkdownNote

## Text to Image / Lumina  (`text_to_image__lumina`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using Lumina.
- example request: "build an image workflow using Lumina"
- description: Generates images from text prompts using NetaYume Lumina, fine-tuned from Neta Lumina for anime-style and illustration generation. | High-quality anime-style image generation with enhanced character understanding and detailed textures. Fine-tuned from Neta Lumina on Danbooru dataset.
- member workflows:
    - image_netayume_lumina_t2i
    - text_to_image_netayume_lumina
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: MarkdownNote, ModelSamplingAuraFlow, PrimitiveStringMultiline (x4), StringConcatenate (x2)
- paired/multiple required: CLIPTextEncode x2
- optional roles: SaveImage
- unresolved nodes: MarkdownNote

## Text to Image / LongCat  (`text_to_image__longcat`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using LongCat.
- example request: "build an image workflow using LongCat"
- description: Generate an image from a text prompt. Input your desired scene description in English or Chinese. Output a photorealistic image with accurate text rendering.
- member workflows:
    - image_longcat_text_to_image
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2), FluxGuidance (x2)
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: CFGNorm, MarkdownNote, ResolutionSelector
- paired/multiple required: CLIPTextEncode x2, FluxGuidance x2
- unresolved nodes: MarkdownNote

## Text to Image / MiniMax  (`text_to_image__minimax`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image from a text prompt using MiniMax.
- example request: "build an image workflow using MiniMax"
- description: Generate a single still image from a text prompt with MiniMax H3, an omni-modal video model used here for its first-frame decode. No file inputs required: the graph samples an empty latent and decodes it through the H3 video VAE. Best suited to a single subject with a defined light source and a simple or blurred background.
- member workflows:
    - image_minimax_h3_t2i
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, CLIPTextEncode
    - latent / canvas: EmptyLatentImage
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: ComfySwitchNode, LTXVSeparateAVLatent, MarkdownNote (x2), ModelAttentionBackend, RandomNoise, ResolutionSelector
- paired/multiple required: MarkdownNote x2
- unresolved nodes: MarkdownNote

## Text to Image / WAN  (`text_to_image__wan`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video using WAN.
- example request: "build a video workflow using WAN"
- description: [Local] intrinsic video decomposition via Wan 2.1 and UniVidX. text prompt -> 4 intrinsic images. Pick this when you need separate lighting and surface data.
- member workflows:
    - UniVidX_ComfyUI__t2RAIN_basic
- node clusters (required structure):
    - sampling: UniVidXSampler
    - output: SaveImage (x4)
    - other operations: Note, UniVidXDecodeIntrinsic, UniVidXLoader, UniVidXTaskMode
- paired/multiple required: SaveImage x4
- unresolved nodes: Note

## Text to Image / WAN VACE  (`text_to_image__wan_vace`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video using WAN VACE.
- example request: "build a video workflow using WAN VACE"
- description: [Local] LDR to HDR via Wan 2.1 VACE and DiffHDR nodes. video -> image. Exports linear HDR as EXR or previews using tonemapping.
- member workflows:
    - ComfyUI-DiffHDR__diffhdr_modular
- node clusters (required structure):
    - inputs: LoadVideo
    - model loading: CLIPLoader, DiffHDRApplyLora, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage
    - other operations: DiffHDRPostprocess, DiffHDRPreprocess, DiffHDRSaveEXR, DiffHDRTonemap, GetVideoComponents, ModelSamplingSD3, TrimVideoLatent, WanVaceToVideo
- paired/multiple required: CLIPTextEncode x2


# Image Edit  (`image_edit`)  -  76 workflow(s), 15 model(s)

## Image Edit / Generic  (`image_edit__generic`)  -  24 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to edit an existing image.
- example request: "build an image workflow"
- description: Edit a photo with a short instruction using MOSS-Image and its edit LoRA: the instruction and the source image are encoded together, so the sampler redraws the same frame and only the described change is applied. Colour, material, outfit and background swaps keep the face, hair, pose and framing intact, and a Compare Images node shows the source next to the result. | Edit an existing image with Ming Image 0.1 Design, a 6B model tuned for UI, infographics, and posters. It takes one input image plus the boxes you draw on it - position, description, and colour palette - and rebuilds them into a layer-by-layer caption, returning one re-rendered image. Ideal for restaging a layout, poster-style re-renders, and turning a photograph into a designed composition. | Edit an image with local open-weights Fibo 2.0 Edit Turbo (int8) and an optional prompt enhancer: a local vision LLM reads the reference image and rewrites your instruction into the structured caption Fibo 2.0 was trained on before TextEncodeFibo2 samples 4 steps at CFG 1. Switch the enhancer off to feed your raw prompt, and compare the result against the source in the built-in image compare node. Ideal for fast instruction-based retouches that keep identity and composition. | Edit images by following natural language instructions with JoyAI-Image-Edit, which leverages multimodal understanding for precise, spatially aware modifications. This workflow takes one reference image and produces one edited image output plus a side-by-side comparison view. Ideal for instruction-guided retouching, region-specific object replacement, and complex scene adjustments requiring relational grounding. | Edit images by following natural language instructions with Mage-Flow-Edit, an instruction-based model that handles background replacement, style changes, localized edits, and restoration using up to three reference images. The workflow takes one input image and produces one edited output plus a side-by-side comparison view. Ideal for creative photo retouching, product image styling, and restoring old or damaged photos. | Edit images using Boogu's instruction-driven model, taking one input image and generating an edited output alongside a side-by-side comparison view. | Edit images using a reference photo and text instructions with the Mage-Flow-Edit Turbo model, a distilled 4B-parameter model that produces results in just 4 steps. It takes 1-3 reference images and a text prompt as input, then outputs an edited image matching your instructions. Ideal for quick style transfers, background replacements, and interactive photo editing sessions. | Edit images with HiDream E1 - Professional natural language image editing model. | Edit images with HiDream E1.1-superior image quality and editing accuracy compared to HiDream-E1-Full. | Edit images with natural language instructions using OmniGen2's advanced image editing capabilities and text rendering support. | Edit images with the Boogu Image Edit model, now optimized with convrot int8 quantization for faster inference and better quality than fp8. This workflow takes 1 input image and produces 1 edited image output. Ideal for quick image edits, batch processing with improved performance, and users wanting to leverage the convrot int8 format's speed and quality advantages. | Generate an image from a text prompt plus a layout drawn in Create Bounding Boxes, using the MOSS-Image base checkpoint. Each box carries its own caption and text boxes render their text. Useful for comparing the base and instruction-tuned checkpoints on the same layout. | Generate an image from a text prompt plus a layout drawn in Create Bounding Boxes. Each box carries its own caption, and text boxes also render their text, so MOSS-Image places every element where you put it. Useful for posters, packaging shots, and any composition where placement matters. | Generate an image from a text prompt with MOSS-Image, an open-weights model that reads a full scene caption and returns one finished image. A prompt upsampler expands a short request into that caption first, so a one-line description is enough. Ideal for editorial photography, product shots, and concept art. | Generate an image from a text prompt with the MOSS-Image base checkpoint, which reads a full scene caption and returns one finished image. The prompt upsampler expands a short request into that caption first. Use it for fine-tuning experiments and comparisons with the instruction-tuned checkpoint. | Generate images by transferring concepts from reference images using SDXL Revision. | Image editing powered by video models' dynamic understanding, creating physically plausible results while preserving character and style consistency. | Input a text prompt and optional reference images. Generate a high-resolution image (up to 2048x2048) with support for text-to-image, image editing, and subject-driven personalization. | Input a text prompt and optionally upload reference images. Generate a high-resolution image up to 2048x2048 with text-to-image, editing, or subject-driven personalization. | Same graph as the Turbo template loaded with the full non-distilled Fibo 2.0 Dev weights: 50 steps at CFG 5 with Bria's structured negative, for the cleanest edits when sampling time is not the constraint. The optional prompt enhancer rewrites the instruction into a structured caption; switch it off to sample the raw prompt. | Upload an image or video and input a text instruction. Generate an edited output using the Capybara model for tasks like style changes, object replacement, or time-of-day adjustments. | Upload your own separate view images, or use one mixed multi-view turnaround sheet. The workflow prepares front, back, left and right views, then reconstructs a textured 3D model with Pixal3D Multi-View. | [Local] Alpha extraction via UniVidX. Image -> Image. Unpacks inputs into intrinsic features and alpha layers for downstream use. | [Local] alpha compositing via UniVidX nodes. input image -> composited image. Extracts foreground with real binary alpha mattes for precise placement on custom backdrops.
- member workflows:
    - 3d_pixal3d_multi_views
    - Image_capybara_v0_1_image_edit
    - UniVidX_ComfyUI__J_alpha_compositing
    - UniVidX_ComfyUI__J_alpha_compositing_api
    - hidream_e1_1
    - hidream_e1_full
    - image_boogu_image_0_1_edit
    - image_boogu_image_0_1_edit_int8
    - image_chrono_edit_14B
    - image_fibo2_edit_dev_pe
    - image_fibo2_edit_turbo_pe
    - image_hidream_o1
    - image_hidream_o1_dev
    - image_joyai_image_edit
    - image_mage_flow_edit_int8
    - image_mage_flow_edit_turbo_int8
    - image_ming_image_01_design_image_edit
    - image_moss_image_base_t2i
    - image_moss_image_base_t2i_bboxes
    - image_moss_image_edit
    - image_moss_image_t2i
    - image_moss_image_t2i_bboxes
    - image_omnigen2_image_edit
    - sdxl_revision_text_prompts
- node clusters (required structure):
    - (none resolved)
- optional roles: RegexExtract, StringFormat, RegexReplace, MarkdownNote, PreviewImage, CreateList, GetItemFromList, ImageCropToMask, ImageCropV2, KSampler, ReferenceLatent, RemoveBackground
- unresolved nodes: MarkdownNote, Note, PrimitiveNode, Reroute

## Image Edit / Qwen Image  (`image_edit__qwen_image`)  -  23 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to edit an existing image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Advanced image editing with multi-image support, improved consistency, and ControlNet integration. | Decompose an image into editable RGBA layers for high-fidelity recolor, replace, resize, and reposition workflows using Qwen-Image-Layered. | Decomposes an image into variable-resolution RGBA layers for independent editing using Qwen-Image-Layered. | Edit images with Qwen Image Edit 2511 Int8, using a quantized model for faster inference while maintaining high quality. Input one image and receive one edited output. Ideal for character-consistent portrait editing, multi-person group photo fusion, and industrial design modifications. | Edit images with Qwen-Image-2.1 by feeding two reference images through a single checkpoint that handles both generation and editing, then comparing the result against the original. Ideal for character-consistent edits, product cutout compositing with real transparency, and style-transfer passes at native 2K | Edit images with Qwen-Image-2.1 by feeding two reference images through a single checkpoint that handles both generation and editing, then comparing the result against the original. Ideal for character-consistent edits, product cutout compositing with real transparency, and style-transfer passes at native 2K. | Edit images with precise bilingual text editing and dual semantic/appearance editing capabilities using Qwen-Image-Edit's 20B MMDiT model. | Edits images from text instructions using Qwen-Image-Edit-2509 with optional Lightning LoRA for few-step sampling. | Edits images via text instructions using Qwen-Image-Edit-2511 with improved character consistency and integrated LoRA. | Generate camera angle variations of your input with an easy to use 3D camera angle node. | Image Edit blueprint | Input an illustration and generate a hyper realistic version using Qwen Image Edit 2509. | Local image editing via QWEN-Image-Edit-2511-Lightning. Up to 3 images (including optional depth/canny control inputs) -> 1 edited image output. Supports text-guided edits with optional structural control. | Relight images using Qwen-Image-Edit with LoRA support. | Remove image backgrounds with Qwen Image 2.1, which natively supports 2K output and RGBA transparency for direct transparent results. It takes 1 input image and returns 1 background-free image plus a comparison view showing the before and after. Ideal for e-commerce product photos, profile pictures, and marketing graphics needing clean cutouts. | Replace materials in objects (e.g., furniture) by combining reference images with Qwen-Image-Edit-2511. | Upload a portrait of your character and a reference lighting image. Generate an image with reference lighting applied and character consistency. | Upload an image and specify a subject. Generate an inflated version of that subject using the INFL8 LoRA for Qwen Image Edit. | Upload an image of your scene and generate multiple views of your input scene with 1 click. | Upload your target image and a reference lighting image. Relights your target image. | Upload your target image and use the trigger phrase "action the [thing]" (e.g., "action the scene"). Generate a toy or action figure version of your subject with articulated joints, plastic materials, and painted details. | [Local] Image edit via Qwen Image and QwenMultiangleCameraNode. 1 img -> 1 img. For adjusting 3D camera angles on existing images. | [Local] image edit via Qwen Image Edit 2511 and QwenMultiangle camera node. One image -> re-angled image. Pick it to rotate a subject's camera angle with the multiple-angles LoRA.
- member workflows:
    - ComfyUI-qwenmultiangle__image_qwen_image_edit_2511_multiangle_camera
    - bepic_templates__bEpic_qwenChangeAngle
    - image-qwen_image_edit_2511_lora_inflation
    - image_edit
    - image_edit_qwen_2509
    - image_edit_qwen_2511
    - image_qwen_image_2_1_background_removal
    - image_qwen_image_2_1_image_edit
    - image_qwen_image_2_1_turbo_image_edit
    - image_qwen_image_edit
    - image_qwen_image_edit_2509
    - image_qwen_image_edit_2509_relight
    - image_qwen_image_edit_2511
    - image_qwen_image_edit_2511_int8
    - image_qwen_image_layered
    - image_to_layers_qwen_image_layered
    - qwen2511_imageEdit
    - template_qwen_image_edit_2511_systms_action
    - template_qwen_multiangle.app
    - templates-1_click_multiple_scene_angles-v1.0
    - templates-image_to_real
    - templates-portrait_light_migration
    - templates_rob_portrait_light_migration.app
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - decoding: VAEDecode
- optional roles: TextEncodeQwenImageEditPlus, CFGNorm, ImageScaleToTotalPixels, KSampler, ModelSamplingAuraFlow, PreviewImage, SaveImage, VAEEncode, CLIPTextEncode, GetNode, MarkdownNote, SetNode
- unresolved nodes: GetNode, Image Load, MarkdownNote, Note, Reroute, SetNode, easy cleanGpuUsed

## Image Edit / Flux  (`image_edit__flux`)  -  6 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image using Flux.
- example request: "build an image workflow using Flux"
- description: Generate images by transferring style from reference images using Flux.1 Redux. | Generate images guided by depth information using Flux.1 LoRA. | Smart image editing that keeps characters consistent, edits specific parts without affecting others, and preserves original styles. | Supports various tasks such as image inpainting, outpainting, and object removal by bytedance-research team | Use reference images to control both style and subject - keep your character's face while changing artistic style, or apply artistic styles to new scenes | [Local] background removal via RMBG nodes with Flux Kontext. Image -> image. Example graph masking a reference latent with AILab_ReferenceLatentMask and Flux Kontext.
- member workflows:
    - ComfyUI-RMBG__ReferenceLatentMask
    - flux1_dev_uso_reference_image_gen
    - flux_depth_lora_example
    - flux_kontext_dev_basic
    - flux_redux_model_example
    - image_flux.1_fill_dev_OneReward
- node clusters (required structure):
    - conditioning: CLIPTextEncode, FluxGuidance
    - decoding: VAEDecode
- optional roles: LoadImage, CLIPVisionEncode, MarkdownNote, USOStyleReference, AILab_ImagePreview, CLIPVisionLoader, CheckpointLoaderSimple, ConditioningZeroOut, DifferentialDiffusion, DualCLIPLoader, InpaintModelConditioning, KSampler
- unresolved nodes: MarkdownNote, Note, PrimitiveNode

## Image Edit / Flux 2 Klein  (`image_edit__flux_2_klein`)  -  6 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to edit an existing image using Flux 2 Klein.
- example request: "build an image workflow using Flux 2 Klein"
- description: A smaller foundation model with exceptional quality-to-size ratio. Ideal for local deployment, fine-tuning on limited hardware, and efficient generation and editing workflows. | BFL undistilled foundation model. Maximum flexibility and control. Great for fine-tuning. | Edits an input image via text instructions using FLUX.2 [klein] 4B. | The fastest variant in the Klein family. Built for interactive applications, real-time previews, and latency-critical production use cases. | Upload reference images and a text prompt. Generate multiple edited images using cached image data for faster processing. | [Local] image editing via Flux. 1 image input -> 1 image output. Performs image editing using the Flux 2 Klein distilled model.
- member workflows:
    - image_edit_flux_2_klein_4b
    - image_flux2_klein_9b_kv_image_edit
    - image_flux2_klein_image_edit_4b_base
    - image_flux2_klein_image_edit_4b_distilled
    - image_flux2_klein_image_edit_9b_base
    - image_flux2_klein_image_edit_9b_distilled
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CFGGuider, CLIPTextEncode
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: EmptyFlux2LatentImage, Flux2Scheduler, GetImageSize, ImageScaleToTotalPixels, RandomNoise, ReferenceLatent (x2)
- paired/multiple required: ReferenceLatent x2
- optional roles: ConditioningZeroOut, LoadImage, MarkdownNote, SaveImage, FluxKVCache
- unresolved nodes: MarkdownNote

## Image Edit / Flux 2  (`image_edit__flux_2`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Flux 2.
- example request: "build an image workflow using Flux 2"
- description: Create product mockups by applying design patterns to packaging, mugs, and other products using multi-reference consistency. | Edits an image from text instructions using Flux.2 [dev], with guidance, schedulers, and optional Turbo LoRAs. | Generate photorealistic images with multi-reference consistency and professional text rendering.
- member workflows:
    - image_edit_flux_2_dev
    - image_flux2
    - image_flux2_fp8
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, CLIPTextEncode, FluxGuidance
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x2), EmptyFlux2LatentImage, Flux2Scheduler, GetImageSize, PrimitiveBoolean, PrimitiveInt (x2), RandomNoise, ReferenceLatent
- optional roles: ImageScaleToTotalPixels, LoadImage, MarkdownNote, SaveImage
- unresolved nodes: MarkdownNote

## Image Edit / FireRed  (`image_edit__firered`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using FireRed.
- example request: "build an image workflow using FireRed"
- description: Edits images via text instructions using FireRed Image Edit 1.1, a diffusion-based instruction-following editing model. | Upload an image and a text prompt to edit it. Generate a modified image with enhanced identity consistency, multi-element fusion, or professional photo restoration.
- member workflows:
    - image_edit_firered_image_edit_1_1
    - image_firered_image_edit1_1
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: TextEncodeQwenImageEditPlus (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: CFGNorm, ComfySwitchNode (x3), ModelSamplingAuraFlow, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x2), ResizeImageMaskNode
- paired/multiple required: TextEncodeQwenImageEditPlus x2
- optional roles: LoadImage, MarkdownNote, SaveImage
- unresolved nodes: MarkdownNote

## Image Edit / Flux 3  (`image_edit__flux_3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using Flux 3.
- example request: "build an image workflow using Flux 3"
- description: Edit an existing image with FLUX 3 Image Klein by pairing one input image with one text prompt, producing a single edited result and a side-by-side comparison view. Ideal for targeted retouching, restyling a photo, or swapping details while keeping the rest of the scene intact. | Edit an existing image with FLUX 3 Image Klein by supplying one input image and two text prompts, producing one refined output alongside a side-by-side comparison view. Ideal for targeted retouching, style adjustments, and detail corrections that preserve the original composition.
- member workflows:
    - image_flux3_klein_distilled_image_edit
    - image_flux3_klein_image_edit
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader (x2), UNETLoader, VAELoader
    - conditioning: CLIPTextEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: BatchImagesNode, ComfySwitchNode, CreateBoundingBoxes, EmptyFlux2LatentImage, Flux3LayoutPrompt, ImageCompare, MarkdownNote (x3), PreviewAny (x2), PrimitiveStringMultiline, ReferenceMultiLatent, TextGenerate
- paired/multiple required: MarkdownNote x3, CLIPLoader x2
- optional roles: ConditioningZeroOut
- unresolved nodes: MarkdownNote

## Image Edit / LongCat  (`image_edit__longcat`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using LongCat.
- example request: "build an image workflow using LongCat"
- description: Edit images using the LongCat-Image-Edit model. Supports bilingual instructions for global, local, text, and reference-guided editing while preserving visual consistency. | Edits images via text instructions using LongCat Image Edit, an instruction-following image editing diffusion model.
- member workflows:
    - image_edit_longcat_image_edit
    - image_longcat_image_edit
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: FluxGuidance (x2), TextEncodeQwenImageEdit (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: FluxKontextMultiReferenceLatentMethod (x2), ImageScaleToTotalPixels
- paired/multiple required: FluxGuidance x2, FluxKontextMultiReferenceLatentMethod x2, TextEncodeQwenImageEdit x2
- optional roles: MarkdownNote, ImageCompare, LoadImage, SaveImage
- unresolved nodes: MarkdownNote

## Image Edit / WAN VACE  (`image_edit__wan_vace`)  -  2 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate an image using WAN VACE.
- example request: "build an image workflow using WAN VACE"
- description: [Local] LDR to HDR reconstruction via WAN VACE and DiffHDR. 1 image -> 1 sRGB image. Recovers extended brightness ranges from flat photos for HDR monitoring or compositing. | [Local] ldr-to-hdr reconstruction via Wan 2.1 VACE. equirectangular image -> linear HDR and sRGB exports. Use this to recover scene lighting from clipped LDR panoramas.
- member workflows:
    - ComfyUI-DiffHDR__diffhdr_hdri
    - ComfyUI-DiffHDR__diffhdr_image
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: UNETLoader, VAELoader
    - output: PreviewImage
    - other operations: DiffHDRSaveEXR, DiffHDRTonemap
- optional roles: DiffHDRPano, DiffHDRVideo

## Image Edit / Bernini  (`image_edit__bernini`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using Bernini.
- example request: "build an image workflow using Bernini"
- description: Edits a single image using a text prompt, leveraging Bernini-R's latent semantic planning for changes like object addition, removal, or style transfer. Ideal for creative edits requiring precise semantic understanding, such as adding a snowman to a scene or altering an object's appearance.
- member workflows:
    - image_edit_bernini_r
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x5), CustomCombo, MarkdownNote, PreviewAny, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, StringConcatenate, StringReplace
- paired/multiple required: CLIPTextEncode x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: MarkdownNote

## Image Edit / Flux Krea  (`image_edit__flux_krea`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Flux Krea.
- example request: "build an image workflow using Flux Krea"
- description: Generate images with the Krea-2 Turbo model while referencing the style of 1-2 uploaded images, using the high-performance Int8 Convrot format for faster, higher-quality inference. This workflow accepts up to two reference image inputs and produces a single stylized output, with optional LoRA support for further creative control. Ideal for matching a specific artistic style in concept art, producing consistent brand visuals, and exploring rapid style transfer for iterative design.
- member workflows:
    - image_krea2_turbo_int8_image_style_reference
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicScheduler, CFGGuider, ConditioningZeroOut, TextEncodeQwenImageEditPlus
    - latent / canvas: EmptyLatentImage (x2)
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: ComfySwitchNode, FluxKontextMultiReferenceLatentMethod, GetImageSize, MarkdownNote, ModelSamplingFlux, PreviewAny, PrimitiveBoolean, PrimitiveStringMultiline (x2), RandomNoise, ResolutionSelector, StringConcatenate, TextGenerate
- paired/multiple required: EmptyLatentImage x2
- unresolved nodes: MarkdownNote

## Image Edit / Lotus  (`image_edit__lotus`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Lotus.
- example request: "build an image workflow using Lotus"
- description: Run Lotus Depth in ComfyUI for zero-shot, efficient monocular depth estimation with high detail retention.
- member workflows:
    - image_lotus_depth_v1_1
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, LotusConditioning
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: DisableNoise, ImageInvert, MarkdownNote, SetFirstSigma
- unresolved nodes: MarkdownNote

## Image Edit / SAM3  (`image_edit__sam3`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to segment an image using SAM3.
- example request: "build an image workflow using SAM3"
- description: Use the SAM3 model to segment the main subject or content from a photo or image, isolating specific objects or regions. Input an image and receive a segmented mask output.
- member workflows:
    - utility_image_segment_sam3
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode
    - output: PreviewImage
    - other operations: InvertMask, JoinImageWithAlpha, MarkdownNote (x2), MaskPreview, SAM3_Detect
- paired/multiple required: MarkdownNote x2
- unresolved nodes: MarkdownNote

## Image Edit / SDPose  (`image_edit__sdpose`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a pose map using SDPose.
- example request: "build an image workflow using SDPose"
- description: Upload an image to detect human poses. Supports detection for both single individuals and multiple people within the same scene.
- member workflows:
    - utility_sdpose_multi_person
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple, UNETLoader
    - output: PreviewImage, SaveImage
    - other operations: DrawBBoxes, ImageBlend, MarkdownNote (x2), PrimitiveInt, RTDETR_detect, ResizeImageMaskNode (x2), SDPoseDrawKeypoints, SDPoseKeypointExtractor
- paired/multiple required: MarkdownNote x2, ResizeImageMaskNode x2
- unresolved nodes: MarkdownNote

## Image Edit / Z-Image  (`image_edit__z_image`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate an image using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: [Local] image-to-image via Z-Image-Turbo. 1 image input + text prompt -> 1 image output. Uses TextEncodeZImageOmni to feed the input image directly into conditioning for high-fidelity i2i edits. Denoise defaults to 0.75 - lower for more structure preservation, higher for more creative freedom.
- member workflows:
    - image_z_image_turbo_i2i
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2), ConditioningZeroOut
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: ModelSamplingAuraFlow
- paired/multiple required: CLIPTextEncode x2


# API / Partner Nodes - Text to Video  (`api_partner_nodes_text_to_video`)  -  70 workflow(s), 9 model(s)

## API / Partner Nodes - Text to Video / Generic  (`api_partner_nodes_text_to_video__generic`)  -  44 workflow(s)  -  source: mixed
- execution: api (API nodes: ByteDance2DraftToFinalVideoNode, ByteDance2ReferenceNode, ByteDance2ReferenceNodeV2, ByteDance2TextToVideoNode, ByteDanceCreateImageAsset, ByteDanceCreateVideoAsset, ByteDanceTextToVideoNode, GeminiNanoBanana2V2, GeminiNode, GeminiNodeV3, GrokVideoNode, GrokVideoReferenceNode, HappyHorseReferenceVideoApi, HappyHorseTextToVideoApi, HeyGenAvatarVideoNode, HeyGenCreateAvatarNode, HeyGenReferenceToVideoNode, HeyGenTalkingPhotoNode, KlingVideoNode, LumaRay32ExtendVideoNode, LumaRay32TextToVideoNode, OpenAIGPTImageNodeV2, PixverseTemplateNode, PixverseTextToVideoNode, PixverseV6FusionVideoNode, PixverseV6TextToVideoNode, PrunaTextToVideoNode, SyncTalkingImageNode, VeedFabricNode, Vidu2ReferenceVideoNode, Vidu2TextToVideoNode, Vidu3TextToVideoNode, Vidu4ReferenceVideoNode, ViduReferenceVideoNode, ViduStartEndToVideoNode, ViduTextToVideoNode)
- when to use: Use to generate a video.
- example request: "build a video workflow"
- description: API text-to-video via Seedance 2.0 (ByteDance). Text prompt only -> 1 video output. Generates high-quality video from a text description using the Seedance 2.0 model. | Create an AI avatar video from a text prompt or uploaded image using HeyGen's avatar generation technology, outputting a single talking-head video. You can save the avatar ID for reuse in future videos. Ideal for personalized digital presenters, social media content creation, and multilingual video production without recording talent. | Create complete audio-visual stories from text with synchronized voices, music, and sound effects. | Create reference-driven videos with PixVerse V6, generating clips that follow the style, motion, and subject traits of your uploaded reference footage. Ideal for matching a specific visual identity across variations, extending a character or scene's look to new actions, and producing consistent branded content from a single source clip. | Create smooth video transitions between defined start and end frames with natural motion interpolation and consistent visual quality. | Generate 30-second cinematic videos from text prompts with Seedance 2.5, producing synchronized sound and music alongside the footage. Input a single text prompt to create one video with precise camera control and multi-shot storytelling. Ideal for brand films, episodic series, and e-commerce content where quality matters more than budget. | Generate a 5-15 second clip with synchronized dialogue and sound from a text prompt using HeyGen Video 1.0, at 480p or 768p in any of six aspect ratios, with no input files required. One video is output. Ideal for UGC-style talking clips, product explainers, and short social ads where the spoken line and the ambience have to match the scene.
 | Generate a 5-15 second video with dialogue and sound from a prompt plus up to nine images, three videos and three audio clips using HeyGen Video 1.0, addressed in the prompt as @Image1, @Video1 and @Audio1. One video is output, with the people, products and places you supply driving the shot. Ideal for product demos, UGC-style clips that keep one presenter or product consistent, and placing your product into a scene of your own. | Generate a reference-based video from a single input image using Seedance 2.0, producing native 4K output with 10-bit color depth and clean fine detail retention. The workflow takes 1 image and outputs 1 video. Ideal for high-resolution motion graphics, cinematic VFX shots requiring downstream grading, and professional video production where 4K finishing is essential. | Generate a talking-head video from a single face photo and an audio clip, with lips and lower-face motion precisely synchronized to the speech. This process outputs one video from one image and one audio input, keeping hands, products, and background completely unchanged. Ideal for product demos, e-commerce introductions, and marketing videos where a static product-holding photo is paired with a voiceover. | Generate a talking-head video from a single photograph using HeyGen's Avatar IV engine, which produces a fully animated video with lip-synced speech, facial expressions, head movements, and automatic gestures. Ideal for creating digital avatars for presentations, virtual characters for interactive content, and personalized video messages from a single image. | Generate a video of up to 15 seconds from a text prompt with Grok Imagine Video 1.5 Lite, the distilled and faster version of Imagine Video 1.5, with native audio and 480p, 720p or 1080p output in any of seven aspect ratios. No input files are required and one video is output. Ideal for social clips, ad mockups, and fast iteration where a full-quality render is not worth the wait.
 | Generate a video with synchronized audio from a text prompt using HappyHorse 1.1, producing a single video output with dialogue, sound effects, and music baked in. Ideal for short episodic series, e-commerce commercials, and game cutscenes. | Generate an 8-panel storyboard from a simple text prompt, then use that storyboard as the foundation for Seedance 2.0. The storyboard provides a visual blueprint for the sequence, allowing you to guide the video with more control over the action, camera angles, composition, and overall storytelling. | Generate cinematic 1080P video clips from text descriptions using Seedance 2.5, supporting clips up to 30 seconds with synchronized sound, music, and dialogue across any aspect ratio. Ideal for short comedy scenes, product showcases, and creative storytelling. | Generate cinematic video from two reference images using Seedance 2.0 Mini, producing a single video output with AI-driven camera motion and consistent character generation. Ideal for storytelling, branded content, and short film creation where shot composition and visual continuity are essential. | Generate cinematic videos from reference images and text prompts. Preserve subject identity and composition while adding expressive motion with synchronized audio. Control camera movement and lighting through detailed prompts. | Generate cinematic videos from text prompts with Dreamina's Seedance 2.0 Mini, supporting AI camera controls and consistent character generation across scenes. This workflow takes only prompt input and produces 1 video output. Ideal for AI filmmaking, branded content creation, and multi-shot storytelling projects. | Generate cinematic videos from text prompts with synchronized audio, controlled camera motion, and stable visuals. Input text descriptions to output high-quality video sequences. | Generate cinematic-quality videos from text prompts using Luma Ray 3.2, with optional video extension by providing a generation ID from a previous run. Ideal for creating short film sequences, marketing content, and iterative video refinement. | Generate draft preview videos from reference images using Seedance 2.5's two-phase draft mode: stage one turns your references and prompt into a fast, low-cost 480p preview plus a reusable draft task ID, and stage two renders the 1080p final from that ID. Ideal for locking in a character or storyboard reference before committing to a full-quality render. | Generate draft preview videos from text prompts using Seedance 2.5's two-phase draft mode: stage one turns your prompt into a fast, low-cost 480p preview plus a reusable draft task ID, and stage two renders the 1080p final from that ID. Ideal for storyboard testing, rapid creative iteration, and budget-conscious production workflows. | Generate expressive videos from two reference images using Vidu Q4 Preview, a flagship model built for vivid character performance, dynamic camera language, and striking visual effects. Ideal for character-driven storytelling, cinematic action sequences, and effect-heavy scenes with explosions, smoke, or particles. | Generate high-quality 1080p videos from text prompts with Vidu Q2 model | Generate high-quality 1080p videos from text prompts with adjustable movement amplitude and duration control using Vidu's advanced AI model. | Generate high-quality videos directly from text prompts using ByteDance's Seedance model. Supports multiple resolutions and aspect ratios with natural motion and cinematic quality. | Generate high-quality, realistic videos from text prompts with fluid motion and rich detail. Input text descriptions to produce dynamic videos with accurate semantic comprehension. | Generate reference-driven videos with Seedance 2.5 by supplying up to 20 images, 6 video clips, and 6 audio clips to control character consistency, visual style, and motion rhythm in a single 4-30 second output with synchronized sound and lip-synced dialogue. Ideal for e-commerce product showcases, brand advertising, and character-driven short series. | Generate video from reference images using HappyHorse 1.1, locking character and scene appearance while following a text prompt for motion. This workflow takes 2 input images (character reference and scene reference) and produces 1 video output. Ideal for short films, advertising, and game cutscenes requiring consistent character design and controllable narrative. | Generate videos from a single text prompt with Pruna P-Video-2, producing 1-20 second clips up to 1080p at 48 fps with native audio, including ambience, sound effects, and lip-synced dialogue from an optional last-frame image. Ideal for narrative shorts, motion posters, ad B-roll, and dialogue shots. | Generate videos that preserve subject characters from multiple reference images using text prompts. Input up to 9 reference images and a scene description to produce a 3-15 second video output. | Generate videos up to 15 seconds at 1080p from a text prompt with PixVerse V6 Max, the higher-fidelity tier of PixVerse V6, with native audio and multi-shot direction in a single generation. No input files required. Ideal for social ads, product concept clips and short-form storytelling. | Generate videos up to 15 seconds at 1080p from text prompts with PixVerse V6, supporting multiple aspect ratios, optional integrated audio, and multi-shot scene direction in a single generation. Ideal for social media ads, product concept testing, and short-form storytelling where narrative continuity and distribution-ready framing matter. | Generate videos with accurate prompt interpretation and stunning video dynamics. | Generate videos with consistent subjects using multiple reference images (up to 7) for character and style continuity across the video sequence. | Generate videos with reference-based consistency for up to 3 subjects, maintaining character and style continuity across the video sequence with cinematic camera movements. | Input a text prompt or upload an image to generate a 1-16 second video with synchronized audio and automatic scene transitions. | Input text prompts or an initial image frame to generate up to 15-second videos with synchronized audio using the Grok model. | Swap products into a reference scene while preserving composition and lighting. Generates a final product image and an automatic video prompt for creating polished product videos. | Turn a still photo and a speech track into a talking, lip-synced video with VEED Fabric 1.0. The face in the image speaks the audio, the clip keeps the image's aspect ratio and runs as long as the audio, up to 300 seconds at 480p or 720p. Ideal for avatars, mascots and localized spokesperson clips driven by your own voice recording or a TTS node. | Upload a reference image and generate a video using the Grok API, supporting up to 6 reference images for enhanced consistency. | Upload a reference video of a real person for identity verification. Use your own photo to validate and generate a personalized video output. | Upload an image and optional text prompt to generate a video with native audio, realistic motion, and high-quality output up to 720p resolution. | Upload two reference images and input a short text description. Generate a detailed Seedance 2.0-style prompt automatically with an LLM helper.
- member workflows:
    - api_bytedance_seedance1_5_text_to_video
    - api_bytedance_text_to_video
    - api_grok_imagine_video_1_5
    - api_grok_imagine_video_1_5_lite_t2v
    - api_grok_reference_to_video
    - api_grok_video
    - api_happyhorse1_0_r2v
    - api_happyhorse1_0_t2v
    - api_happyhorse1_1_r2v
    - api_happyhorse1_1_t2v
    - api_heygen_avatar_video
    - api_heygen_talking_photo
    - api_heygen_video_1_r2v
    - api_heygen_video_1_t2v
    - api_luma_ray3_3_t2v
    - api_pixverse6_r2v
    - api_pixverse6_t2v
    - api_pixverse_t2v
    - api_pixverse_v6_max_t2v
    - api_pruna_p_video_t2v
    - api_seedance2_0_mini_r2v
    - api_seedance2_0_mini_t2v
    - api_seedance2_0_r2v
    - api_seedance2_0_r2v_4k
    - api_seedance2_0_r2v_real_human
    - api_seedance2_0_t2v
    - api_seedance2_5_draft_r2v
    - api_seedance2_5_draft_t2v
    - api_seedance2_5_r2v
    - api_seedance2_5_t2v
    - api_seedance2_5_t2v_1080p
    - api_seedance2_t2v
    - api_sync_so_talking_image
    - api_veed_fabric_1_0
    - api_vidu_q2_r2v
    - api_vidu_q2_t2v
    - api_vidu_q3_text_to_video
    - api_vidu_q4_preview_r2v
    - api_vidu_reference_to_video
    - api_vidu_start_end_to_video
    - api_vidu_text_to_video
    - template_product_placement
    - template_seedance2_storyboard_to_video
    - template_seedance_2_0_plus_llm_prompt_helper
- node clusters (required structure):
    - (none resolved)
- optional roles: LoadImage, MarkdownNote, GeminiNodeV3, SaveText, SaveVideo, BatchImagesNode, ByteDance2DraftToFinalVideoNode, ByteDance2ReferenceNode, ByteDance2ReferenceNodeV2, ByteDance2TextToVideoNode, ByteDanceCreateImageAsset, ByteDanceCreateVideoAsset
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Video / Kling  (`api_partner_nodes_text_to_video__kling`)  -  7 workflow(s)  -  source: mixed
- execution: api (API nodes: Kling4ReferenceToVideoNode, Kling4TextToVideoNode, KlingOmniProTextToVideoNode, KlingTextToVideoWithAudio, KlingVideoNode)
- when to use: Use to generate a video from a text prompt using Kling.
- example request: "build a video workflow using Kling"
- description: API multi-shot storyboard video via Kling 3.0 (kling-v3). 1 input image (start frame, LoadImage node) -> 1 video output (VHS_VideoCombine). Generates 1-6 sequential shots in a single generation: each shot has its own text prompt (max 512 chars) and duration set directly on the KlingVideoNode. Use for storyboards, scene sequences, and narrative clips with multiple camera cuts. Prompts go into multi_shot.storyboard_N_prompt inputs; multi_shot must match shot count exactly (e.g. '3 storyboards'). Aspect ratio defaults to 16:9, resolution to 720p - override only on explicit user request. | Bring your stories to life with videos featuring synchronized dialogue, music, sound effects, and ambient audio from text prompts. | Generate a ten-second cinematic clip from text alone with Kling 4.0, holding one single continuous shot through a full high-speed chase while also producing native stereo audio. Ideal for short-form spectacle, concept pitches, and motion references where a single unbroken take carries the idea. | Generate videos from text descriptions using Kling O1. Create dynamic video content with natural language prompts. | Hold a character and a location from two reference images and shoot a three-shot fashion fragment between them: an extreme close-up with a snap zoom, a mid-hall turn with a whip-pan, and a 120-degree orbit under the arched window. Each cut lands on a new position inside the same room, with two hard cuts in five seconds and stereo audio generated for the space. | Input text or image prompts to generate 15-second video sequences with multi-shot compositions, complex camera movements, and consistent subjects. | Upload text, audio, or images to generate a 15-second video with synchronized audio, character consistency, and storyboard control.
- member workflows:
    - Kling3_multiShot
    - api_kling2_6_t2v
    - api_kling4_r2v
    - api_kling4_t2v
    - api_kling_o3_t2v
    - api_kling_omni_t2v
    - api_kling_v3_video
- node clusters (required structure):
    - (none resolved)
- optional roles: LoadImage, GetVideoComponents, Kling4ReferenceToVideoNode, Kling4TextToVideoNode, KlingOmniProTextToVideoNode, KlingTextToVideoWithAudio, KlingVideoNode, MarkdownNote, SaveVideo, VHS_VideoCombine
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Video / MiniMax  (`api_partner_nodes_text_to_video__minimax`)  -  5 workflow(s)  -  source: official
- execution: api (API nodes: MinimaxHailuo03ContextIRNode, MinimaxHailuo03ReferenceNode, MinimaxHailuo03RegenerateNode, MinimaxHailuo03TextToVideoNode)
- when to use: Use to generate a video from a text prompt using MiniMax.
- example request: "build a video workflow using MiniMax"
- description: Generate 5-15 second videos from text prompts using MiniMax H3 Max Turbo, a high-speed text-to-video model optimized for rapid inference. This workflow takes no file inputs and outputs a single video file, delivering 768p clips in roughly 1-3 seconds of generation time. Ideal for quick social media content, iterative creative exploration, and fast prototyping of video concepts. | Generate polished, prompt-faithful video from text using MiniMax H3 Max, producing a single finished video clip with no input files required. Ideal for rapid concept exploration, cinematic text-to-video scenes, and high-throughput creative iteration where speed and visual quality matter. | Generate video directly from text using the MiniMax H3 model, producing a 5-15 second clip with native stereo audio. Ideal for rapid concept visualization, short-form social media content, and pre-visualization for film or advertising projects. | Generate video from a reference image or set of multimodal inputs using MiniMax H3, which interprets text, images, video, and audio together to produce coherent 5-15 second clips with native stereo audio. Input up to 2 images for first/last frame control, or up to 9 images, 3 video clips, and 3 audio clips in Omni Reference mode, and output a single video in your chosen aspect ratio. Ideal for commercial content creation, character-driven storytelling, and rapid visual concept validation. | Generate video from an image reference using MiniMax H3 Max, which animates your still frame while preserving its composition, visual identity, and aspect ratio. The model generates faster than real time with strong prompt adherence and polished cinematic motion. Ideal for bringing character art to life, animating product shots, and creating video from concept images.
- member workflows:
    - api_minimax_h3_max_r2v
    - api_minimax_h3_max_t2v
    - api_minimax_h3_max_turbo_t2v
    - api_minimax_h3_r2v
    - api_minimax_h3_t2v
- node clusters (required structure):
    - output: SaveVideo
- optional roles: MarkdownNote, LoadImage, MinimaxHailuo03ContextIRNode, MinimaxHailuo03ReferenceNode, MinimaxHailuo03RegenerateNode, MinimaxHailuo03TextToVideoNode, SaveText
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Video / WAN  (`api_partner_nodes_text_to_video__wan`)  -  5 workflow(s)  -  source: official
- execution: api (API nodes: Wan2ReferenceVideoApi, Wan2TextToVideoApi, Wan3ReferenceToVideoApi, WanTextToVideoApi)
- when to use: Use to generate a video from a text prompt using WAN.
- example request: "build a video workflow using WAN"
- description: Generate a video from two reference images using Wan3.0's reference-to-video capability, which produces up to 30 seconds of footage with character consistency and lifelike visuals. Ideal for creating consistent character animations, scene replication from visual references, and producing audiovisual content from static imagery. | Generate videos from text prompts using Wan 2.7. Supports audio reference to create lip-synced video | Generate videos up to 30 seconds directly from text prompts using Wan3.0-Video, producing a single video output with lifelike visuals and synchronized sound. This workflow transforms written descriptions into immersive, production-ready footage with strong character consistency and cinematic quality. Ideal for cinematic storytelling, character-driven animations, and quick multimedia content creation. | Generate videos with synchronized audio, enhanced motion, and superior quality. | Upload a reference video and character image to generate a video with consistent character features.
- member workflows:
    - api_wan2_7_r2v
    - api_wan2_7_t2v
    - api_wan3_0_r2v
    - api_wan3_0_t2v
    - api_wan_text_to_video
- node clusters (required structure):
    - output: SaveVideo
- optional roles: LoadImage, MarkdownNote, LoadAudio, RecordAudio, Wan2ReferenceVideoApi, Wan2TextToVideoApi, Wan3ReferenceToVideoApi, WanTextToVideoApi
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Video / Gemini  (`api_partner_nodes_text_to_video__gemini`)  -  3 workflow(s)  -  source: official
- execution: api (API nodes: GeminiVideoOmniV2)
- when to use: Use to generate a video from a text prompt using Gemini.
- example request: "build a video workflow using Gemini"
- description: Generate cinematic video from natural language prompts using Gemini Omni Flash, transforming text descriptions into a single video output with world-aware motion, lighting, and sound. Ideal for social media content creation, rapid video prototyping, and iterative visual storytelling with conversational editing. | Generate cinematic videos with audio from text prompts or reference images using Gemini Omni 1.1 Flash, supporting aspect ratio, resolution, and keyframe interpolation controls. Ideal for text-to-video storytelling, animating product shots or illustrations, and iterative conversational video editing. | Generate video that incorporates specific subjects from multiple reference images using Gemini Omni 1.1 Flash's reference mode: characters, products, and objects from your images appear in the scene while the image itself is never shown as a frame. Bind each image to a role with tags like <IMAGE_REF_0>, combine character references with style references, and direct the scene in plain language with native audio. Ideal for character-driven stories, product placement, and brand-consistent content.
- member workflows:
    - api_google_gemini_omni_flash_1_1_r2v
    - api_google_gemini_omni_flash_1_1_t2v
    - api_google_gemini_omni_flash_t2v
- node clusters (required structure):
    - output: SaveVideo
    - other operations: GeminiVideoOmniV2
- optional roles: LoadImage, MarkdownNote
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Video / Flux 3  (`api_partner_nodes_text_to_video__flux_3`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Flux3MaxTextToVideoNode, Flux3TextToVideoNode)
- when to use: Use to generate a video from a text prompt using Flux 3.
- example request: "build a video workflow using Flux 3"
- description: Generate a video with synchronized audio from a single text prompt with FLUX 3 Max, which reasons about and expands the prompt before rendering. Outputs one clip of 5 to 20 seconds at 720p or 1080p, at any aspect ratio from 9:16 to 21:9, with ambient sound, speech, and effects created alongside the picture and no input files required. Ideal for cinematic concept clips, ad mockups, and social videos that need matching sound design. | Generate full videos from a single text prompt with FLUX 3, which expands natural language into complete scenes with synchronized audio, no reference images or complex settings needed. Outputs 1 video in 720p at 24fps, 5 to 20 seconds long, supporting any aspect ratio from 9:16 to 21:9 with built-in sound design. Ideal for quick social media clips, ad mockups, and concept films requiring realistic motion and cinematic style.
- member workflows:
    - api_bfl_flux3_t2v
    - api_flux3_max_t2v
- node clusters (required structure):
    - output: SaveVideo
- optional roles: Flux3MaxTextToVideoNode, Flux3TextToVideoNode

## API / Partner Nodes - Text to Video / LTX-2  (`api_partner_nodes_text_to_video__ltx_2`)  -  2 workflow(s)  -  source: mixed
- execution: api (API nodes: LtxApi25TextToVideo)
- when to use: Use to generate a video from a text prompt using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate video from text with LTX 2.5, a video model built for fast, high-quality production work. Type a prompt and get a cinematic clip with sharp detail, accurate motion, and support for native multishot scenes that hold character and style across cuts.
- member workflows:
    - api_ltx2_5_t2v
    - api_ltxv_text_to_video
- node clusters (required structure):
    - (none resolved)
- optional roles: GetVideoComponents, LtxApi25TextToVideo, LtxvApiTextToVideo, SaveVideo, VHS_VideoCombine
- unresolved nodes: LtxvApiTextToVideo

## API / Partner Nodes - Text to Video / Veo  (`api_partner_nodes_text_to_video__veo`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: Veo3VideoGenerationNode)
- when to use: Use to produce a video using Veo.
- example request: "build a video workflow using Veo"
- description: Produce a video using Veo. Structurally it applies a sequence of node operations. Boundary inputs: IMAGE; outputs: VIDEO.
- member workflows:
    - api_veo3
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
    - other operations: Veo3VideoGenerationNode

## API / Partner Nodes - Text to Video / WAN 2.6  (`api_partner_nodes_text_to_video__wan_2_6`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: WanTextToVideoApi)
- when to use: Use to generate a video from a text prompt using WAN 2.6.
- example request: "build a video workflow using WAN 2.6"
- description: API text-to-video via Wan 2.6. Text prompt only -> 1 video output. Generates 1080P video with enhanced quality, smoother motion, and improved prompt understanding.
- member workflows:
    - api_wan2_6_t2v
- node clusters (required structure):
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, WanTextToVideoApi


# Image Tools  (`image_tools`)  -  47 workflow(s), 4 model(s)

## Image Tools / Generic  (`image_tools__generic`)  -  38 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image.
- example request: "build an image workflow"
- description: Adds a glow/bloom effect around bright image areas via GPU fragment shader. | Adds lens-style chromatic aberration (color fringing) using a real-time GPU fragment shader. | Adds procedural film grain texture for a cinematic look via GPU fragment shader. | Adjusts black point, white point, and gamma for tonal range control via GPU shader. | Adjusts hue, saturation, and lightness of an image using a real-time GPU fragment shader. | Adjusts image brightness and contrast using a real-time GPU fragment shader. | Adjusts saturation, temperature, tint, and vibrance using a real-time GPU fragment shader. | Applies Gaussian, Box, or Radial blur to soften images and create stylized depth or motion effects. | Applies bilateral (edge-preserving) blur to soften images while retaining detail. | Balances colors across shadows, midtones, and highlights using a real-time GPU fragment shader. | Combine four input images into a 2x2 grid. Automatically resizes mismatched images and outputs a single 2048x2048 image. | Enhances edge contrast via unsharp masking for a sharper image appearance. | Fine-tunes tone and color with per-channel curve adjustments using a real-time GPU fragment shader. | Input your text prompt and optionally an image, audio, or video. Generate text output with configurable reasoning, coding, and multilingual support. | Interactive walkthrough of core image and mask nodes - create and refine masks, combine overlapping masks with different operations, and composite images using feathered masks. No models required. | Manipulates individual RGBA channels for masking, compositing, and channel effects. | Sharpens image details using a GPU fragment shader for enhanced clarity. | Side-by-side demo of built-in blueprint color-grading subgraphs. | Splits an image into a 2×2 grid of four equal tiles. | Splits an image into a 3×3 grid of nine equal tiles. | Splits an image into a configurable columns×rows grid of equal tiles for tiled generation or processing. | [Local] Background removal via Florence2 and Finegrain Box Segmenter. Image -> cutout image. Use when a Florence-2 prompt locates an object and Finegrain boxes it for masking. | [Local] Background removal via YOLO and ComfyUI-RMBG nodes. Image -> image and mask. Example graph for YOLO segmentation and background removal. | [Local] PBR material estimation via Chord ChordLoadModel and ChordMaterialEstimation. Image path -> material maps. Estimates material maps from a texture image with the Chord model. | [Local] PBR material estimation via Chord model. Texture image -> material maps. Estimates PBR material maps including normals and height from generated textures. | [Local] Region inpainting via SEGSDetailer and SAM. 1 image -> 2 images. Automatically refines selected areas through localized inpainting rather than global regeneration. | [Local] background removal and object cropping via ComfyUI-RMBG SegmentV2, CropObject, ImageCompare. Image -> cutout image and comparison preview. Pick for segmentation demos with mask-based cropping. | [Local] background removal via RMBG node. Image -> image and mask previews. Example workflow from ComfyUI-RMBG for trying removal with mask preview. | [Local] background removal via RMBG with AILab canvas. Images -> cutout images. Use for interactive AILab canvas background removal and preview in the ComfyUI-RMBG pack. | [Local] background removal via RMBG. AILab_LoadImage image -> image. Example graph for the ComfyUI-RMBG pack that loads an image and runs RMBG removal. | [Local] background removal via RMBG. image -> image. Pick this for quick single-image background isolation using the dedicated node. | [Local] background removal via SAM2Segment and AILab_MaskEnhancer. Image plus mask -> cut-out image. Pick it to segment objects into a mask with SAM2 inside ComfyUI-RMBG. | [Local] detailer hook preview via PreviewDetailerHookProvider. image -> image. Use it to monitor intermediate enhancement steps before pasting results. | [Local] face detailing via MaskDetailerPipe. image -> image. Automates facial enhancement and iterative upscaling passes. | [Local] image load and preview via AILab_LoadImage, AILab_LoadImageSimple, AILab_LoadImageAdvanced and AILab_Preview/AILab_ImagePreview. Image files -> loaded image and on-screen preview. Minimal RMBG example graph showing image loading and preview nodes, with no background removal model attached. | [Local] image stitch via ComfyUI-RMBG AILab_ImageStitch. Two images -> combined image. Example graph for the pack's ImageStitch node, not background removal. | [Local] matting via SAM2Segment and AILab_SDMatte. Image -> matte image. Matte and refine cutouts after SAM2 segmentation; pick over plain background removal for soft edges. | [Local] segmentation and captioning via Florence-2 in ComfyUI-RMBG. Image in -> previewed image/text out. Use for Florence-2 object detection, captioning or segmentation rather than plain RMBG background removal.
- member workflows:
    - ComfyUI-Chord__chord_image_to_material
    - ComfyUI-Impact-Pack__2-MaskDetailer
    - ComfyUI-Impact-Pack__3-SEGSDetailer
    - ComfyUI-Impact-Pack__5-PreviewDetailerHookProvider
    - ComfyUI-RMBG__AILab_canvas
    - ComfyUI-RMBG__ComfyUI-RMBG_V2.4.0_new_nodes
    - ComfyUI-RMBG__Florence2_Finegrain_box
    - ComfyUI-RMBG__LoadImage_Node
    - ComfyUI-RMBG__V2.7.0_ImageStitch
    - ComfyUI-RMBG__V2.7.0_LoadImage
    - ComfyUI-RMBG__V2.8.0_sam2
    - ComfyUI-RMBG__V2.9.0_Matte_matting
    - ComfyUI-RMBG__YOLO_Node
    - ComfyUI-RMBG__florence2_node
    - ComfyUI-RMBG__image_mask_preview
    - RemoveBackground_RMBG2
    - basic_image_color_adjustment
    - basic_mask_operations_and_compositing
    - bepic_templates__bEpic_chord_simple
    - brightness_and_contrast
    - chromatic_aberration
    - color_adjustment
    - color_balance
    - color_curves
    - crop_images_2x2
    - crop_images_3x3
    - edge_preserving_blur
    - film_grain
    - glow
    - hue_and_saturation
    - image_blur
    - image_channels
    - image_levels
    - llm_gemma4_text_gen
    - sharpen
    - split_image_grid_to_tiles
    - unsharp_mask
    - utility_image_stitch
- node clusters (required structure):
    - (none resolved)
- optional roles: MaskPreview, PreviewImage, GLSLShader, ImageCropV2, MarkdownNote, MaskComposite, CustomCombo, ailab_ImagePreview, AILab_LoadImage, SaveImage, AILab_ImagePreview, CurveEditor
- unresolved nodes: Label (rgthree), MarkdownNote, Note, Reroute, ailab_ImagePreview, ailab_MaskPreview, workflow>Impact::MAKE_BASIC_PIPE, workflow>MAKE_BASIC_PIPE

## Image Tools / SAM3  (`image_tools__sam3`)  -  5 workflow(s)  -  source: custom
- execution: local
- when to use: Use to segment an image using SAM3.
- example request: "build an image workflow using SAM3"
- description: [Local] Segment images via SAM3. Image -> Image. Generate segmentation masks using text prompts. | [Local] background removal and segmentation via RMBG, SAM3, YOLOv8 and Florence-2 nodes. Images -> masks and cut-out images. Use to strip backgrounds or segment objects, faces and clothing in the ComfyUI-RMBG example graph. | [Local] image segmentation via SAM3. 1 input image -> 1 output image. Picks point prompts for precise boundary control instead of boxes or text. | [Local] image segmentation via SAM3. image -> image. Uses text prompts, boxes, or points to isolate subjects in images. | [Local] interactive segmentation via SAM3. image -> mask. Pick this when you need precise control using text, boxes, or click points.
- member workflows:
    - ComfyUI-RMBG__V3.0.0_nodes
    - ComfyUI-SAM3__image_seg_point
    - ComfyUI-SAM3__image_seg_text
    - ComfyUI-SAM3__scene_seg
    - ComfyUI-SAM3__scene_seg_interactive
- node clusters (required structure):
    - (none resolved)
- optional roles: AILab_Preview, AILab_MaskPreview, AILab_ColorToMask, AILab_Florence2, AILab_Florence2ToCoordinates, AILab_ImageCompare, AILab_ImageMaskToList, AILab_ImageResize, AILab_ImageToList, AILab_LoadImage, AILab_MaskToList, AILab_YoloV8
- unresolved nodes: AILab_Florence2ToCoordinates, MultibandFromBatch, MultibandPreview, MultibandResize, MultibandSave, Reroute, SAM3Segment

## Image Tools / BiRefNet  (`image_tools__birefnet`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to remove the background from an image using BiRefNet.
- example request: "build an image workflow using BiRefNet"
- description: Removes or replaces image backgrounds using BiRefNet segmentation and alpha compositing. | [Local] background removal and segmentation via ComfyUI-RMBG RMBG and BiRefNet nodes. Load image in, segment clothes, body, face and fashion, stitch or combine -> masked images. Sample workflow for RMBG pack's segmenter nodes.
- member workflows:
    - ComfyUI-RMBG__Comfyu-rmbg_v2.9.3_node_sample
    - remove_background_birefnet
- node clusters (required structure):
    - (none resolved)
- optional roles: AILab_ICLoRAConcat, AILab_ImageCombiner, AILab_ImageMaskConvert, AILab_ImagePreview, AILab_ImageStitch, AILab_LamaRemover, AILab_LoadImage, AILab_LoadImageAdvanced, AILab_LoadImageSimple, AILab_MaskCombiner, AILab_MaskEnhancer, AILab_MaskExtractor

## Image Tools / Qwen Image  (`image_tools__qwen_image`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to caption an image as text using Qwen Image.
- example request: "build a text workflow using Qwen Image"
- description: Generate text with visual understanding using Qwen3-VL models, processing one input image to produce contextual text responses. Ideal for image captioning, visual question answering, and multimodal content generation. | Use the Qwen3.5 model to analyze an input image and generate descriptive text prompts. This workflow performs image captioning and reverse prompt engineering.
- member workflows:
    - llm_qwen3_5_text_gen
    - llm_qwen3vl_text_gen
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader
    - other operations: MarkdownNote, PreviewAny, TextGenerate
- unresolved nodes: MarkdownNote


# API / Partner Nodes - Text to Image  (`api_partner_nodes_text_to_image`)  -  44 workflow(s), 12 model(s)

## API / Partner Nodes - Text to Image / Generic  (`api_partner_nodes_text_to_image__generic`)  -  21 workflow(s)  -  source: mixed
- execution: api (API nodes: BriaImageGenerate, ElevenLabsSpeechToText, GrokImageNode, MetaMuseImageTextToImageApi, OpenAIGPTImageNodeV2, OpenRouterImageNode, QuiverTextToSVGNodeV2, RecraftColorRGB, RecraftControls, RecraftTextToVectorNode, RecraftV4TextToImageNode, RecraftV4TextToVectorNode)
- when to use: Use to generate an image from a text prompt.
- example request: "build an image workflow"
- description: Create campaign-ready posters, product shots, and multilingual signage using OpenAI's latest model. Upload an optional reference image, write a detailed prompt, and generate high-quality, photorealistic outputs. | GPT Image 2.5 Flare is a fast image generation and editing template optimized for rapid prototyping and creative iteration. It delivers quick previews for concept exploration, mood boards, poster drafts, character sketches and scene mockups. Balances solid visual quality with low latency, ideal for high-volume trial-and-error creative workflows. | GPT Image 2.5 Sunburst is a high-fidelity image editing and generation template for creating polished, detail-rich visuals. It preserves subject identity, complex materials, lighting consistency and scene structure across iterations. Suitable for concept refinement, product visuals, character scenes, cinematic backgrounds and finished artwork that requires precision, realism and visual depth. | Generate a clean, flat vector logo from a text prompt using Recraft V4.1, producing one scalable SVG output with no file input required. Ideal for branding and logo design, vector illustration projects, and creating consistent flat-color graphics for presentations or merchandise. | Generate an image from a text prompt with Bria FIBO 2.0, one model for both generation and editing. It is trained on structured VGL captions, so a detailed poster-style brief is followed closely, exact wording is rendered literally, and the node also returns that structured prompt so the same image can be rebuilt or refined from JSON. Ideal for editorial covers, posters and typography-led layouts. | Generate editable SVG vector graphics directly from text prompts using Arrow 2, producing one image output with no file inputs required. Ideal for illustrations, technical diagrams, logo design, and turning written briefs into clean, scalable vector assets. | Generate enhanced images with improved details, text rendering, and creative control using the advanced quality mode in Grok Imagine. | Generate high-quality images from text prompts using Microsoft's MAI Image 2.6, a top-tier model excelling in legible in-image text rendering, photorealistic portraits, and commercial product visuals. Simply enter a prompt to produce 1 image output with no file inputs. Ideal for marketing assets, cinematic concept art, and creative projects requiring polished, text-accurate visuals. | Generate high-quality vector images from text prompts using Recraft's AI vector generator. | Generate images from text prompts using Grok Imagine Image 2.0, an Aurora-engine model that excels at crisp short-text rendering and 1K or 2K outputs across 14 aspect ratios. Enter one text prompt and receive one generated image, with low or medium quality options to control cost. Ideal for poster and cover design, product and brand visuals, and character-driven scenes where consistency matters. | Generate images from text prompts using Meta's Muse Image 1.0 model, which follows instructions faithfully, edits with precision, and composes from multiple references. The workflow takes no file inputs and produces 1 image output, with a Save Image Advanced node handling the final render. Ideal for creative concept exploration, rapid visual prototyping, and producing polished marketing or social media imagery. | Generate images from text prompts using OpenAI GPT Image 1 API. | Generate images from text prompts using Recraft V4.1 Flash, a fast, low-cost model that produces one image output with no file inputs required. Ideal for high-volume generation, rapid iteration, photography and portraits, atmospheric scenes, and quick logo concept exploration. | Generate images from text prompts with Microsoft's MAI Image 2.6 Flash, a fast, cost-efficient model that emphasizes low latency and high throughput for rapid visual ideation. The workflow takes a single text prompt and produces one image output. Ideal for concept sketches, marketing mockups, and batch asset drafts where speed matters more than final polish. | Generate photorealistic and illustrative images from text prompts using Recraft V4.1, producing one high-quality image per generation. This model excels at interpreting short prompts with creative flair, delivering natural photorealism, expressive vectors, and clean illustrations. Ideal for concept art, logo and typography design, and lifestyle product mockups. | Input a text prompt and optional reference images to generate one or more SVG vector graphics. | Input a text prompt to generate a high-resolution image. Use the Recraft V4 model for enhanced realism, text accuracy, and detailed vector graphics. Output a 1024x1024 or 2048x2048 image based on your selected version. | Input text prompts to generate clean, editable vector SVG graphics. Create logos, icons, or illustrations with structured paths suitable for design tools. | Input text prompts to generate high-quality images quickly using the Grok model. | Upload an audio or video file to transcribe its speech into accurate, editable text. | [API] text-to-image via gpt-image-2. prompt -> image. Relies on OpenAI's cloud generation instead of local models.
- member workflows:
    - api_bria_fibo2_t2i
    - api_elevenLabs_speech_to_text
    - api_grok_imagine_image_2_t2i
    - api_grok_imagine_image_quality_image_generation
    - api_grok_text_to_image
    - api_meta_muse_image_t2i
    - api_openai_gpt_image_25_flare_t2i
    - api_openai_gpt_image_25_sunburst_t2i
    - api_openai_gpt_image_2_t2i
    - api_openai_image_1_t2i
    - api_openrouter_mai_image_26_flash
    - api_openrouter_mai_image_26_t2i
    - api_quiver_arrow2_text_to_svg
    - api_quiver_text_to_svg
    - api_recraft_v4_1_flash_t2i
    - api_recraft_v4_1_t2i
    - api_recraft_v4_1_text_to_vector
    - api_recraft_v4_t2i
    - api_recraft_v4_text_to_vector
    - api_recraft_vector_gen
    - api_t2i_OpenAi_GPT2
- node clusters (required structure):
    - (none resolved)
- optional roles: RecraftColorRGB, BriaImageGenerate, ElevenLabsSpeechToText, GrokImageNode, LoadAudio, MarkdownNote, MetaMuseImageTextToImageApi, OpenAIGPTImageNodeV2, OpenRouterImageNode, QuiverTextToSVGNodeV2, RecordAudio, RecraftControls
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Image / Nano-Banana  (`api_partner_nodes_text_to_image__nano_banana`)  -  5 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiImage2Node, GeminiNanoBanana2V2)
- when to use: Use to generate an image from a text prompt using Nano-Banana, Gemini.
- example request: "build an image workflow using Nano-Banana"
- description: API / cloud generation via Nano Banana 2. text input -> 1 image output. | Generate images from text descriptions using the ultra-fast Gemini 3.1 Flash-Lite Image model, designed for rapid creation and iteration with no file inputs required. This workflow produces one or more generated images based on your text prompt, with support for interleaved text and image inputs. Ideal for quick concept visualization, rapid prototyping, and iterative design exploration. | Generate images from text prompts using Nano Banana 2.1, Google's fast Gemini-based image model, delivering one finished image with no input files required. Ideal for brand marketing posters, e-commerce product visuals, and IP character series. | Generate images from text prompts using the Nano Banana 2 model, producing one image output with no file inputs required. Ideal for rapid prototyping, concept art generation, and creative exploration. | [API] text-to-image via Nano-Banana. text -> image. Uses Gemini APIs for cloud-based generation without local GPU requirements.
- member workflows:
    - NanoBanana2_text_to_image
    - api_google_nano_banana2_text_to_image
    - api_nano_banana_2_1_t2i
    - api_nano_banana_2_lite_t2i
    - api_t2i_nanoBananaPro
- node clusters (required structure):
    - (none resolved)
- optional roles: GeminiImage2Node, GeminiNanoBanana2V2, SaveImage, SaveImageAdvanced

## API / Partner Nodes - Text to Image / Ideogram  (`api_partner_nodes_text_to_image__ideogram`)  -  4 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiNodeV3, IdeogramPImage, IdeogramTextToImageApi, IdeogramV3, IdeogramV4)
- when to use: Use to generate an image from a text prompt using Ideogram.
- example request: "build an image workflow using Ideogram"
- description: Generate images from text descriptions using Ideogram's fast P-Image model, which delivers quality comparable to leading image models at significantly lower cost and latency. One text prompt is accepted and one image is output, with controls for quality level and resolution. Ideal for rapid iteration in design workflows, high-volume content production, and cost-sensitive A/B testing of visual concepts. | Generate images from text prompts using Ideogram 4.5, with controls for canvas size, quality level, automatic prompt rewriting, and seed. No file inputs are required and one image is output. Ideal for typography-led posters, layout-driven design work, marketing visuals, and fast exploration of visual concepts.
 | Input a text prompt in structured JSON format. Generate a high-quality image with precise layout control, text rendering, and color palette support.
- member workflows:
    - api_ideogram_p_image_t2i
    - api_ideogram_v3_t2i
    - api_ideogram_v4_5_t2i
    - api_ideogram_v4_t2i
- node clusters (required structure):
    - (none resolved)
- optional roles: BuildJsonPromptIdeogram, CreateBoundingBoxes, GeminiNodeV3, IdeogramPImage, IdeogramTextToImageApi, IdeogramV3, IdeogramV4, MarkdownNote, SaveImage, SaveImageAdvanced
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Image / Seedream  (`api_partner_nodes_text_to_image__seedream`)  -  3 workflow(s)  -  source: official
- execution: api (API nodes: ByteDanceSeedreamNodeV3)
- when to use: Use to generate an image from a text prompt using Seedream.
- example request: "build an image workflow using Seedream"
- description: Generate high-quality images from text prompts using Seedream 5.0 Flash, producing detailed 1K, 1.5K, or 2K visuals with cinematic lighting and accurate text rendering, saved as a single image output. | Generate images from text prompts using Seedream 5.0 Pro, producing one high-quality output. Ideal for creative concept art, marketing visual mockups, and rapid ideation for design projects. | Input your text prompt to generate a detailed image. This workflow creates a visual representation of your description with precise control over style and layout.
- member workflows:
    - api_bytedance_seedream_5_0_flash_t2i
    - api_bytedance_seedream_5_0_lite_t2i
    - api_bytedance_seedream_5_0_pro_t2i
- node clusters (required structure):
    - other operations: ByteDanceSeedreamNodeV3
- optional roles: SaveImage, SaveImageAdvanced

## API / Partner Nodes - Text to Image / Flux Krea  (`api_partner_nodes_text_to_image__flux_krea`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Krea2ImageNode, Krea3ImageNode)
- when to use: Use to generate an image from a text prompt using Flux Krea.
- example request: "build an image workflow using Flux Krea"
- description: Generate images from text prompts using Krea 3, producing a single finished image per run with no input files required. Ideal for e-commerce product staging, fashion and portrait work, and architectural concept rendering. | Input a text prompt describing your desired scene. Generate four aesthetic, high-quality images that interpret your prompt with creative variety.
- member workflows:
    - api_krea2_t2i
    - api_krea3_t2i
- node clusters (required structure):
    - (none resolved)
- optional roles: Krea2ImageNode, Krea3ImageNode, SaveImage, SaveImageAdvanced

## API / Partner Nodes - Text to Image / Kling  (`api_partner_nodes_text_to_image__kling`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: KlingImageGenerationNode, KlingOmniProImageNode)
- when to use: Use to generate an image from a text prompt using Kling.
- example request: "build an image workflow using Kling"
- description: Generate an image from a text prompt using the Kling 3.0 model. Input your descriptive text to receive a high-quality, prompt-accurate visual output. | Generate high-quality images from text prompts using Kling O1, with support for up to 10 reference images to guide style and content. Features flexible aspect ratios and 1K/2K resolution options.
- member workflows:
    - api_kling_omni_image
    - api_kling_v3_t2i
- node clusters (required structure):
    - output: SaveImage
- optional roles: KlingImageGenerationNode, KlingOmniProImageNode

## API / Partner Nodes - Text to Image / Qwen Image  (`api_partner_nodes_text_to_image__qwen_image`)  -  2 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiImageNode, QwenImageTextToImageApi)
- when to use: Use to generate an image from a text prompt using Qwen Image, Gemini.
- example request: "build an image workflow using Qwen Image"
- description: Generate detailed, production-ready images from text prompts using the Qwen Image 3.0 Pro model, supporting up to 4.5k tokens for complex layouts like posters, infographics, and storyboards. It renders crisp 10px text, realistic textures, and native support for 12 languages and over 100 art styles, with optional 1-3 reference images for fusion editing. Ideal for marketing materials, educational diagrams, and UI design mockups. | [API] image edit via Qwen Image Edit and Gemini. 3 images + text -> 2 images. Combines local Qwen Image Edit 2511 sampling with a Gemini image node.
- member workflows:
    - api_qwen3_t2i
    - bepic_templates__bEpic_IE_QwenGemini
- node clusters (required structure):
    - output: SaveImage
- optional roles: GetNode, Image Load, SetNode, CFGNorm, ModelSamplingAuraFlow, TextEncodeQwenImageEditPlus, bEpicGetPath, CLIPLoader, EmptyLatentImage, GeminiImageNode, ImageBatch, KSampler
- unresolved nodes: GetNode, Image Load, MarkdownNote, SetNode

## API / Partner Nodes - Text to Image / Flux 2 Klein  (`api_partner_nodes_text_to_image__flux_2_klein`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: ByteDanceSeedreamNode, Flux2ProImageNode, FluxKontextMaxImageNode, FluxKontextProImageNode, GeminiImage2Node, GeminiImageNode)
- when to use: Use to generate an image using Flux 2 Klein, Qwen Image.
- example request: "build an image workflow using Flux 2 Klein"
- description: [Local] image generation via Qwen Image, Flux 2 Klein and hosted image APIs. text or reference image -> 9 images. Example bEpic pack graph mixing local qwen image edit, flux 2 klein and API image models with a floating viewer.
- member workflows:
    - bepic_templates__bEpic_imageCreation
- node clusters (required structure):
    - model loading: CLIPLoader (x2), LoraLoader, UNETLoader (x2), VAELoader (x2)
    - conditioning: CFGGuider, CLIPTextEncode (x2), TextEncodeQwenImageEditPlus (x2)
    - latent / canvas: EmptyLatentImage, VAEEncode
    - sampling: KSampler, KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode (x2)
    - output: PreviewImage, SaveImage (x8)
    - other operations: ByteDanceSeedreamNode, CFGNorm (x2), EmptyFlux2LatentImage, Flux2ProImageNode, Flux2Scheduler, FluxKontextMaxImageNode, FluxKontextProImageNode, GeminiImage2Node, GeminiImageNode, GetNode (x22), Image Load, ImageScaleToTotalPixels, ModelSamplingAuraFlow (x2), PrimitiveStringMultiline (x2), RandomNoise, ReferenceLatent (x2), SetNode (x8), bEpicGetPath (x8), bEpicSendToViewer, bepicSetPath, easy cleanGpuUsed
- paired/multiple required: GetNode x22, SaveImage x8, SetNode x8, bEpicGetPath x8, CFGNorm x2, CLIPLoader x2, CLIPTextEncode x2, ModelSamplingAuraFlow x2, ReferenceLatent x2, TextEncodeQwenImageEditPlus x2, UNETLoader x2, VAEDecode x2, VAELoader x2
- unresolved nodes: GetNode, Image Load, SetNode, easy cleanGpuUsed

## API / Partner Nodes - Text to Image / Flux 3  (`api_partner_nodes_text_to_image__flux_3`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: Flux3ImageNode)
- when to use: Use to generate an image from a text prompt using Flux 3.
- example request: "build an image workflow using Flux 3"
- description: Generate an image from a text prompt with FLUX 3 Image, which interprets and expands the prompt before generating it, at any of sixteen aspect ratios and up to 4K. Web and image search grounding is on by default and no input files are required. One image is output. Ideal for typography-led posters, layout-driven design work, and fast exploration of visual concepts.

- member workflows:
    - api_bfl_flux3_t2i
- node clusters (required structure):
    - output: SaveImageAdvanced
    - other operations: Flux3ImageNode

## API / Partner Nodes - Text to Image / Hunyuan3D  (`api_partner_nodes_text_to_image__hunyuan3d`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: HunyuanImageTextToImageApi)
- when to use: Use to generate an image from a text prompt using Hunyuan3D.
- example request: "build an image workflow using Hunyuan3D"
- description: Tencent Hunyuan's preview image model, strengthening three areas over previous versions: multilingual text rendering and typesetting for advertising, infographics and UI design; photorealism and stylistic range covering cinematic, comic, commercial photography and illustration; and editing consistency that preserves key traits of people and products across background, outfit and style changes. Outputs up to 4K resolution.
- member workflows:
    - api_tencent_hy_image_3_5_preview_t2i
- node clusters (required structure):
    - output: SaveImageAdvanced
    - other operations: HunyuanImageTextToImageApi

## API / Partner Nodes - Text to Image / WAN  (`api_partner_nodes_text_to_image__wan`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: WanTextToImageApi)
- when to use: Use to generate an image from a text prompt using WAN.
- example request: "build an image workflow using WAN"
- description: Generate images with excellent prompt following and visual quality using Wan 2.5.
- member workflows:
    - api_wan_text_to_image
- node clusters (required structure):
    - output: SaveImage
    - other operations: MarkdownNote, WanTextToImageApi
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Text to Image / Z-Image  (`api_partner_nodes_text_to_image__z_image`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: OpenRouterLLMNode)
- when to use: Use to generate an image from a text prompt using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: [API] text-to-image via Z-Image and AgentYPython. text prompt -> image. Uses agentY sidebar chat to generate images with Z-Image turbo.
- member workflows:
    - story_refs_only
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x3)
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler (x2)
    - decoding: VAEDecode (x2)
    - other operations: AgentYPython (x2), ModelSamplingAuraFlow, OpenRouterLLMNode (x2), PrimitiveStringMultiline
- paired/multiple required: CLIPTextEncode x3, AgentYPython x2, KSampler x2, OpenRouterLLMNode x2, VAEDecode x2


# Image to Video  (`image_to_video`)  -  40 workflow(s), 8 model(s)

## Image to Video / LTX-2  (`image_to_video__ltx_2`)  -  10 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from an input image using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Apply the ltx2.3-transition LoRA to create smooth style and scene transitions. This workflow uses a specialized model to blend visual elements. | Generate personalized videos with synchronized audio from a text prompt, reference image, and short audio clip. Use ID-LoRA to adapt a person's appearance and voice in a single generative model. | Generate short video clips that stay visually consistent with a reference sheet inventorying characters, props, and locations. Input a composite reference image and a split prompt describing the action, outputting a single video clip. Ideal for character-driven animation, consistent prop visualization, and location-specific scene generation. | Generate video from a single image using LTX-2.5, producing a high-fidelity clip with industry-leading pixel quality and native multishot capability that holds character, environment, and lighting across connected scenes. Ideal for cinematic production, VFX shot creation, and content pipelines requiring fast, locally deployable video generation. | Generates video from a single input image using LTX-2.3. | Transform static images into dynamic videos with synchronized audio-video generation using LTX-2 distilled model. Optimized for faster generation while maintaining quality. Features expressive lip sync, natural motion, and improved speed. | Transform static images into dynamic videos with synchronized audio-video generation using LTX-2. Features expressive lip sync, natural motion, and efficient performance. | Upload an image and an audio file to generate a high-quality video with synchronized lip movements using LTX-2.3. | Upload any image to apply the Squish It LoRA effect. Generate a squished animation using the LTX-2 I2V model.
- member workflows:
    - image_to_video_ltx_2_3
    - template_ltx2_3_ic_lora_ingredients
    - template_ltx2_3_style_transition
    - video_ltx2_3_i2v
    - video_ltx2_3_ia2v
    - video_ltx2_3_id_lora
    - video_ltx2_5_i2v
    - video_ltx2_i2v
    - video_ltx2_i2v_distilled
    - video_ltx2_i2v_lora
- node clusters (required structure):
    - conditioning: CLIPTextEncode, LTXVConditioning
    - decoding: LTXVAudioVAEDecode
    - other operations: CreateVideo, EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVSeparateAVLatent, PrimitiveInt
- optional roles: LoraLoaderModelOnly, MarkdownNote, CFGGuider, CLIPLoader, KSamplerSelect, LTXVAddGuide, LTXVDualCFGGuider, LTXVImgToVideoInplace, LTXVPreprocess, LoadImage, ManualSigmas, RandomNoise
- unresolved nodes: MarkdownNote, Reroute

## Image to Video / Generic  (`image_to_video__generic`)  -  7 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from an input image.
- example request: "build a video workflow"
- description: A lightweight 2B model that generates videos from English and Russian prompts with high visual quality. | Generate synchronized video with matching audio from a text prompt plus a reference image using the FastH3 OmniRef eight-step distilled model, which renders both streams in just eight sampling steps. Takes 1 input image and produces 1 video output. Ideal for animating reference characters, creating talking avatars, and rapid social media video prototyping. | Generate videos based on audio, image, and text, keeping the character's lip sync. | Generate videos by first creating images from text prompts. | Input a text prompt and select a model checkpoint. Generate a high-quality video using Causal Forcing or Causal Forcing++ with 1 to 4 inference steps. | Upload an image and provide a text instruction. Generate a new video where the input is edited according to your instruction, preserving structure and temporal coherence. | Upload an image or video and a text instruction to perform generation or editing tasks. Generate a new or edited image or video based on the provided instruction.
- member workflows:
    - txt_to_image_to_video
    - video_capybara_v0_1_image_to_video
    - video_capybara_v0_1_video_edit
    - video_causal_forcing_i2v
    - video_fasth3_ominiref_r2v
    - video_humo
    - video_kandinsky5_i2v
- node clusters (required structure):
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, MarkdownNote
- optional roles: CLIPTextEncode, KSampler, VAELoader, ARVideoI2V, AudioEncoderEncode, AudioEncoderLoader, BasicGuider, BasicScheduler, BlockSparseAttention, CFGGuider, CLIPLoader, CLIPVisionEncode
- unresolved nodes: MarkdownNote

## Image to Video / WAN  (`image_to_video__wan`)  -  7 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video using WAN.
- example request: "build a video workflow using WAN"
- description: Generate dynamic videos with cinematic camera movements using Wan 2.1 Fun Camera 1.3B model. | Generate high-quality videos with advanced camera control using the full 14B model | Generate videos from a single image using Wan-Move, with fine-grained point-level motion control via trajectory guidance. | Generate videos from images using Wan 2.1. | Generate videos from start and end frames using Wan 2.1 inpainting. | Trajectory-controlled Video Generation. | Upload a source Image and target audio. Generate a full-body dubbed video with synchronized motion while preserving the original identity, background, and camera movement.
- member workflows:
    - image_to_video_wan
    - video_wan2.1_fun_camera_v1.1_1.3B
    - video_wan2.1_fun_camera_v1.1_14B
    - video_wan2_1_infinitetalk
    - video_wan_ati
    - video_wanmove_480p
    - wan2.1_fun_inp
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, MarkdownNote
- paired/multiple required: CLIPTextEncode x2
- optional roles: AudioEncoderEncode, GenerateTracks, AudioEncoderLoader, BasicScheduler, CFGGuider, ConditioningZeroOut, KSamplerSelect, LoadAudio, LoraLoaderModelOnly, ModelPatchLoader, Painter, RandomNoise
- unresolved nodes: MarkdownNote, Reroute

## Image to Video / WAN 2.2  (`image_to_video__wan_2_2`)  -  7 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from an input image using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Fast text-to-video and image-to-video generation with 5B parameters. Optimized for rapid prototyping and creative exploration. | Generate minute-scale coherent dance videos from music using a hierarchical Wan Dancer framework. Input a music file and reference image to produce high-resolution dance videos with global structure and temporal continuity. | Image to Video blueprint | Image-to-video with Wan 2.2 using a start image plus text prompt to extend motion from the still frame. | Transform static images and audio into dynamic videos with perfect synchronization and minute-level generation. | Transform static images into dynamic videos with precise motion control and style preservation using Wan 2.2.
- member workflows:
    - image_to_video
    - image_to_video_wan_2_2
    - video_wan2_2_14B_fun_camera
    - video_wan2_2_14B_i2v
    - video_wan2_2_14B_s2v
    - video_wan2_2_5B_ti2v
    - video_wan_dancer
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, ModelSamplingSD3
- paired/multiple required: CLIPTextEncode x2
- optional roles: KSampler, LatentConcat, MarkdownNote, WanSoundImageToVideoExtend, Note, CustomCombo, AudioEncoderEncode, AudioEncoderLoader, BasicScheduler, CLIPVisionEncode, ImageFromBatch, KSamplerAdvanced
- unresolved nodes: MarkdownNote, Note

## Image to Video / MiniMax  (`image_to_video__minimax`)  -  5 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from an input image using MiniMax.
- example request: "build a video workflow using MiniMax"
- description: Generate a new video by feeding MiniMax H3 a reference image, up to nine images, three videos, and three audio clips alongside a text prompt, letting the model match character identity, motion, camera, voice, or style from those sources while producing native stereo audio in a single pass. Output is a single MP4 video at up to 2K resolution and 24fps for roughly 15 seconds. Ideal for character-consistent animation, style-transfer video generation, and audio-synced content creation. | Generate video from multiple still images with MiniMax H3, anchoring up to four reference frames anywhere along the timeline to control motion and composition. Ideal for creating video continuations from existing clips, forcing narrative beats at specific moments, and producing footage with synchronized native audio for cinematic or social media content. | Generate video with synchronized native audio from a single still image using MiniMax H3 and the FastH3 8-step checkpoint. One image produces one video output with matching voice, sound effects, and music in a single pass. Ideal for animating portraits, bringing product shots to life, and quick social media clips. | Start with an image and an empty prompt, then describe the motion, shots, and audio you want MiniMax H3 to generate. The workflow produces one video with synchronized audio and remains fully editable before execution. | Turn a starting image into a short, high-quality video clip with MiniMax H3, a unified multimodal model that generates visuals and synchronized audio in a single pass. The workflow takes 1 input image and produces 1 video output, with optional text or reference media to steer motion, style, or sound. Ideal for rapid concept visualization, character-driven animation tests, and creating shareable social media clips with built-in sound design.
- member workflows:
    - video_fastvideo_fasth3_i2v
    - video_minimax_h3_i2v
    - video_minimax_h3_i2v_continuation
    - video_minimax_h3_multiframe_reference
    - video_minimax_h3_r2v
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, UNETLoader, VAELoader (x2)
    - conditioning: BasicGuider, BasicScheduler
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode, VAEDecodeAudio
    - output: SaveVideo
    - other operations: ComfyMathExpression, CreateVideo, MarkdownNote (x2), PrimitiveFloat, RandomNoise
- paired/multiple required: MarkdownNote x2, VAELoader x2
- optional roles: MiniMaxH3AddGuide, BlockSparseAttention, ImageScaleToTotalPixels, LoraLoaderModelOnly, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift, ModelAttentionBackend, ResolutionSelector
- unresolved nodes: MarkdownNote

## Image to Video / Hunyuan Video  (`image_to_video__hunyuan_video`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from an input image using Hunyuan Video.
- example request: "build a video workflow using Hunyuan Video"
- description: Animate still images into dynamic videos with precise motion and camera control. Maintains visual consistency while bringing photos and illustrations to life with smooth, natural movements. | [Local] image-to-video via LeapfusionHunyuanI2VPatcher. image -> video. Use this for direct photo-to-clip animation using Kijai's native HunyanVideo sampler.
- member workflows:
    - ComfyUI-KJNodes__leapfusion_hunyuuanvideo_i2v_native_testing
    - video_hunyuan_video_1.5_720p_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: BasicScheduler, CLIPTextEncode
    - sampling: KSamplerSelect
    - decoding: VAEDecodeTiled
    - other operations: ModelSamplingSD3, Note
- optional roles: CFGGuider, MarkdownNote, SamplerCustomAdvanced, CreateVideo, EasyCache, RandomNoise, SaveVideo, VAEDecode, CLIPVisionEncode, CLIPVisionLoader, ConditioningZeroOut, DisableNoise
- unresolved nodes: MarkdownNote, Note

## Image to Video / Bernini  (`image_to_video__bernini`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing image using Bernini.
- example request: "build an image workflow using Bernini"
- description: Generate an edited image with matched lighting and view a side-by-side before/after comparison. Ideal for portrait and product relighting, consistent lighting across photo sets, and e-commerce catalog photography.
- member workflows:
    - video_bernini_r_image_editing
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: BatchImagesNode, ComfySwitchNode (x5), CustomCombo, GetImageSize, ImageCompare, MarkdownNote (x4), PreviewAny, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, ResizeImageMaskNode, StringConcatenate, StringReplace
- paired/multiple required: MarkdownNote x4, CLIPTextEncode x2, LoadImage x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: MarkdownNote

## Image to Video / LTX-Video  (`image_to_video__ltx_video`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from an input image using LTX-Video.
- example request: "build a video workflow using LTX-Video"
- description: Generate videos from still images.
- member workflows:
    - ltxv_image_to_video
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2), LTXVConditioning
    - sampling: KSamplerSelect, SamplerCustom
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, LTXVImgToVideo, LTXVScheduler, MarkdownNote, Note
- paired/multiple required: CLIPTextEncode x2
- unresolved nodes: MarkdownNote, Note


# API / Partner Nodes - Video to Video  (`api_partner_nodes_video_to_video`)  -  39 workflow(s), 7 model(s)

## API / Partner Nodes - Video to Video / Generic  (`api_partner_nodes_video_to_video__generic`)  -  23 workflow(s)  -  source: mixed
- execution: api (API nodes: BeebleSwitchXVideoEdit, BriaRemoveVideoBackground, BriaTransparentVideoBackground, BriaVideoGreenScreen, BriaVideoReplaceBackground, ByteDance2ReferenceNode, ByteDance2ReferenceNodeV2, ByteDanceSeedreamNode, GrokVideoEditNode, GrokVideoExtendNode, HappyHorseVideoEditApi, HeyGenVideoTranslateNode, HiggsfieldMotionTransferNode, LumaRay32ExtendVideoNode, LumaRay32VideoEditNode, MinimaxHailuoVideoNode, RunwayAleph2KeyframeNode, RunwayAleph2PromptImageNode, RunwayAleph2VideoToVideoNode, SyncLipSyncNode, VeedLipsyncNode, VeedRemoveVideoBackgroundNode, ViduExtendVideoNode)
- when to use: Use to generate a video.
- example request: "build a video workflow"
- description: API reference-to-video via Seedance 2.0 (ByteDance). 1 reference image + 1 reference video -> 1 video output. Generates, edits, or extends video using multimodal references for subject consistency, video editing, and video extension. | Cut the subject out of a video with VEED Remove Video Background: the node segments the foreground in every frame and returns the frames plus an alpha mask, and Join Image with Alpha turns that mask into a transparent PNG sequence ready to composite. Choose standard, fast or green-screen keying; clips up to 60 seconds. | Edit a video using text instructions and optional reference images for style transfer or local replacement. Takes a video and up to 5 reference images as input, outputs an edited video. | Edit existing video clips with Seedance 2.5 by describing changes in natural language, enabling local object removal, background swaps, scene replacements, and style transfers while preserving the original composition and motion. Feed in 1 video and receive 1 edited video output with temporally consistent results across frames. Ideal for e-commerce background replacement, stylized remakes of footage, and ad re-editing with targeted retouching. | Edit videos using natural language commands with Luma Ray 3.2, processing one input video and generating an edited output based on your textual instructions. | Extend an existing video forward or backward by generating new segments that seamlessly preserve the original characters, scenes, and aspect ratio. The source video must be 1.9-30 seconds long, and you specify the duration of the new clip (4-30 seconds, default 5), with output delivered as MP4 or MOV. Ideal for continuing unfinished footage, creating extended narrative sequences, and building longer edit chains. | Generate a single edited video that preserves original motion and timing while applying your chosen changes. Ideal for portrait and product relighting, consistent lighting across photo sets, and e-commerce apparel presentation. | Re-sync a speaker's mouth in footage you already shot to a new speech track with VEED Lipsync 2.0. The person on camera speaks the new audio while the rest of the shot stays as recorded; the output keeps the video's resolution at 25 fps and runs as long as the audio, up to 600 seconds. If the clip is shorter than the speech it plays forward and backward until the audio ends. Ideal for dubbing a line, swapping a script without re-shooting, or making localized versions of existing footage. | Recast an existing video with Higgsfield Genjutsu: the source clip keeps its motion, camera movement, timing and audio, while the cast, location and style are rebuilt from 1 to 8 reference images. Describe which reference replaces what in the optional prompt. Output length follows the source clip. Ideal for restyling footage, swapping the cast of a shot, and building scenes a shoot never covered. | Synchronize a video's lip movements with a new audio track using the Sync Labs sync-3 model, which automatically adjusts the speaker's mouth to match your uploaded audio. This workflow takes 1 video and 1 audio input to produce 1 synced video output. Ideal for dubbing foreign-language content, correcting mismatched dialogue in video production, or animating a static portrait to speak. | Take a picture with your phone, upload it and generate a studio grade product video. | Translate video content into another language with full lip-sync using HeyGen Translate, which automatically transcribes speech, translates it, generates a cloned voice, and re-syncs mouth movements. Takes 1 input video and produces 1 output video. Ideal for localizing talking-head videos, dubbing multilingual presentations, and creating lip-synced content for global audiences. | Upload a short video and a text prompt describing the next scene. Generate a seamless video extension using Grok extend. | Upload a source video and a reference image, then provide a prompt describing the desired environment and lighting. Generate an edited video where masked regions follow your prompt and reference, while unmasked regions preserve the original pixels and motion, plus an alpha matte for compositing. | Upload a video and a background image. Detect the main subject and replace the original background with the provided image. Ideal for content creation, virtual production, and background cleanup in video projects. | Upload a video and generate its frames with a transparent background, outputting image and mask sequences along with a WebM video supporting an alpha channel. | Upload a video and use the Vidu Q2 model to extend it up to 7 seconds and enhance the resolution to 1080p. | Upload a video to edit. Input your desired modifications. Generate an edited video using the Grok API. | Upload a video to generate a green screen version with a transparent or solid background. | Upload a video to remove its background and replace it with your chosen color. Generate a video with the subject isolated and a new solid background. | Upload a video to remove the background. Generate a new video with a transparent or solid color background.
- member workflows:
    - 0193-spec-0160-renderLightChar-v006
    - api_beeble_switchx_video_edit
    - api_bria_remove_video_background
    - api_bria_remove_video_background_transparent
    - api_bria_video_green_screen
    - api_bria_video_replace_background
    - api_grok_video_edit
    - api_grok_video_extend
    - api_happyhorse1_0_video_edit
    - api_heygen_video_translate
    - api_higgsfield_genjutsu_motion_transfer
    - api_luma_ray3_2_video_edit
    - api_runway_aleph2_video_edit
    - api_seedance2_5_video_editing
    - api_seedance2_5_video_extend
    - api_seedance2_reference2v
    - api_sync_so_lip_sync_video
    - api_veed_lipsync_2_0
    - api_veed_remove_video_background
    - api_vidu_video_extension
    - templates-photo_to_product_vid
    - utility-bria_remove_video_background
    - v2v_api_switchx_videoStyleTransfer
- node clusters (required structure):
    - (none resolved)
- optional roles: SaveVideo, GetVideoComponents, MinimaxHailuoVideoNode, ImageBatch, LoadImage, AudioConcat, BatchImagesNode, BeebleSwitchXVideoEdit, BriaRemoveVideoBackground, BriaTransparentVideoBackground, BriaVideoGreenScreen, BriaVideoReplaceBackground
- unresolved nodes: MarkdownNote, Note

## API / Partner Nodes - Video to Video / Kling  (`api_partner_nodes_video_to_video__kling`)  -  7 workflow(s)  -  source: mixed
- execution: api (API nodes: ByteDance2ReferenceNode, ClaudeNode, GeminiNode, Kling4EditVideoNode, KlingMotionControl, KlingOmniProEditVideoNode, KlingOmniProVideoToVideoNode)
- when to use: Use to edit an existing video using Kling.
- example request: "build a video workflow using Kling"
- description: API video editing via Kling O3. 1 video + 1 reference image -> 1 edited video output. Enables precise subject editing and scene composition with native audio-visual synchronization. | Apply precise character actions and expressions from a reference video to your character image with synchronized motion control. | Edit an existing clip in one pass with Kling 4.0: name the change and list what must survive, and the model rewrites only that part while keeping the subject, costume, shot count, camera work and timing intact. Output inherits the input's length and aspect ratio, and stereo audio is regenerated to match the new scene. Ideal for swapping environments, wardrobe or style on footage that is already cut. | Edit videos with natural language commands, featuring video reference mode for quick generation of high-quality style transfers, element additions, and background modifications. | Transform videos with Kling O1. Edit video content, change styles, or replace characters using text prompts and reference images. | Upload a character image and a motion reference video. Generate a new video where your character performs the actions from the reference, with enhanced facial consistency using optional element binding. | [API] video-to-video via Kling Omni and Seedance 2.0. Image and video in -> video out. bEpic Audi template that retimes and converts FPS on shots then re-renders them through video API models.
- member workflows:
    - api_kling4_video_edit
    - api_kling_motion_control
    - api_kling_motion_control3
    - api_kling_o3_video_edit
    - api_kling_omni_edit_video
    - api_kling_omni_v2v
    - bepic_templates__bEpic_Audi
- node clusters (required structure):
    - (none resolved)
- optional roles: GetNode, SetNode, DisplayAny, bepic_Math, ImpactSwitch, Note, bepic_colorTransform, AILab_UnbatchImages, ClaudeNode, CreateVideo, VHS_LoadVideoPath, bEpicGetPath
- unresolved nodes: DF_Int_to_Float, DisplayAny, GetNode, Int, MarkdownNote, Note, SetNode

## API / Partner Nodes - Video to Video / Gemini  (`api_partner_nodes_video_to_video__gemini`)  -  4 workflow(s)  -  source: official
- execution: api (API nodes: ByteDance2ReferenceNodeV2, GeminiNodeV3, GeminiVideoOmniV2, OpenAIGPTImageNodeV2)
- when to use: Use to generate a video using Gemini.
- example request: "build a video workflow using Gemini"
- description: A one click workflow to recreate an input video with your character. All you need to do is upload the video and your character. This workflow uses Google Gemini LLM to analyze the input video and reverse engineer a prompt. You can prompt for specific details to inject into this analysis. The workflow extracts the first frame of the video and swaps your character into it, sends this into Seedance 2.0 Reference-to-Video and generates | Edit videos through natural language conversation with Gemini Omni 1.1 Flash. Upload a clip (10 seconds or less) and describe the change: swap backgrounds, restyle scenes, add or remove elements, and the model applies the edit while preserving everything you did not mention. Add \"Keep everything else the same\" for maximum consistency; simple prompts work best. Outputs video with native audio at 16:9 or 9:16, with 1080p/4K output available. Ideal for remixes, style changes, and iterative refinement without re-generating from scratch. | Edit videos with natural language using Gemini Omni Flash, transforming a single input video into one edited output based on your descriptive instructions. Specify the duration and aspect ratio in your prompt, and leverage Omni's world knowledge for intuitive edits like changing backgrounds or adding atmosphere. Ideal for quick social media remixes, cinematic scene adjustments, and iterative video refinements without technical prompts. | Extend an existing video by up to 10 seconds per step, building stories up to 40 seconds total with Gemini Omni 1.1 Flash. The model analyzes the last 10 seconds of context, keeping characters, motion, and audio coherent as the scene continues from where it left off. Optionally attach reference images to introduce new characters mid-story. Works on uploaded clips and previously generated videos; describe how the scene should continue in plain language. Ideal for longer narratives, scene branching, and iterative storytelling.
- member workflows:
    - api_google_gemini_omni_flash_1_1_edit
    - api_google_gemini_omni_flash_1_1_extend
    - api_google_gemini_omni_flash_video_edit
    - template_seedance2_0_viral_videos_character_swap
- node clusters (required structure):
    - inputs: LoadVideo
    - output: SaveVideo
- optional roles: CustomCombo, ByteDance2ReferenceNodeV2, GeminiNodeV3, GeminiVideoOmniV2, GetVideoComponents, ImageFromBatch, LoadImage, MarkdownNote, OpenAIGPTImageNodeV2, SaveImage, Video Slice
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Video to Video / Topaz  (`api_partner_nodes_video_to_video__topaz`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: TopazVideoEnhanceV2)
- when to use: Use to upscale / enhance a video using Topaz.
- example request: "build a video workflow using Topaz"
- description: Upload a video to enhance its resolution using the Topaz Starlight Precise 2.5 model. This workflow produces sharper 4K output with reduced artifacts compared to previous versions. | Upscale GenAI video footage with Astra 2's creative diffusion, adding dynamic detail and texture. Adjust creativity, sharpness, and prompt-based guidance for precise stylization of AI-generated content.
- member workflows:
    - api_topaz_astra2
    - api_topaz_starlight_precise25
- node clusters (required structure):
    - inputs: LoadVideo
    - output: SaveVideo
    - other operations: TopazVideoEnhanceV2

## API / Partner Nodes - Video to Video / Flux  (`api_partner_nodes_video_to_video__flux`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: FluxVideoEditNode)
- when to use: Use to edit an existing video using Flux.
- example request: "build a video workflow using Flux"
- description: Edit and transform images with FLUX using simple text prompts. Change characters, objects, backgrounds, styles, or entire scenes while preserving the key details and overall consistency of the original image.
- member workflows:
    - api_flux_video_edit
- node clusters (required structure):
    - inputs: LoadVideo
    - output: SaveVideo
    - other operations: FluxVideoEditNode, MarkdownNote
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Video to Video / WAN  (`api_partner_nodes_video_to_video__wan`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: Wan2VideoEditApi)
- when to use: Use to edit an existing video using WAN.
- example request: "build a video workflow using WAN"
- description: Upload a video and corresponding images to edit and replace characters or scenes using the Wan2.7 model.
- member workflows:
    - api_wan2_7_video_edit
- node clusters (required structure):
    - inputs: LoadImage (x2), LoadVideo
    - output: SaveVideo
    - other operations: Wan2VideoEditApi
- paired/multiple required: LoadImage x2

## API / Partner Nodes - Video to Video / WAN 2.6  (`api_partner_nodes_video_to_video__wan_2_6`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: WanReferenceVideoApi)
- when to use: Use to generate a video using WAN 2.6.
- example request: "build a video workflow using WAN 2.6"
- description: Creating identity-preserved videos with natural movement and cinematic quality.
- member workflows:
    - api_wan_r2v
- node clusters (required structure):
    - inputs: LoadVideo (x2)
    - output: SaveVideo
    - other operations: WanReferenceVideoApi
- paired/multiple required: LoadVideo x2


# API / Partner Nodes - 3D  (`api_partner_nodes_3d`)  -  36 workflow(s), 3 model(s)

## API / Partner Nodes - 3D / Generic  (`api_partner_nodes_3d__generic`)  -  20 workflow(s)  -  source: official
- execution: api (API nodes: GeminiImage2Node, RecraftRemoveBackgroundNode, Rodin3D_Detail, Rodin3D_Gen2, Rodin3D_Gen25_Image, Rodin3D_Gen25_Text, Rodin3D_Regular, Rodin3D_Sketch, Rodin3D_Smooth, TripoConversionNode, TripoImageToModelNode, TripoImageToModelNodeV2, TripoMultiviewToModelNode, TripoP1ImageToModelNode, TripoP1MultiviewToModelNode, TripoP1TextToModelNode, TripoPSeriesImageToModelNode, TripoPSeriesMultiviewToModelNode, TripoPSeriesTextToModelNode, TripoRetargetNode, TripoRigNode, TripoTextToModelNode, TripoTextToModelNodeV2, TripoTextureNode, TripoTextureNodeV2)
- when to use: Use to generate a 3D model.
- example request: "build a 3d workflow"
- description: Build 3D models from multiple angles with Tripo's advanced scanner. | Craft 3D objects from descriptions with Tripo's text-driven modeling. | Generate a production-ready 3D model from a single reference image using Tripo P2.0 Preview, which builds native quad topology with clean edge flow suited for rigging and animation. It takes 1 input image, with optional multi-view references for better side and back accuracy. Ideal for game characters and props, hard-surface mechanical assets, and high-volume real-time pipelines. | Generate detailed 3D models from single photos using Rodin AI. | Generate detailed 4X mesh quality 3D models from photos using Rodin Gen2 | Generate precise 3D models from text with Tripo 3.0's ultra-high resolution geometry and realistic PBR materials. | Generate production-ready 3D hero assets with high-density geometry and PBR materials. Takes text prompts or reference images as input and outputs detailed 3D meshes suitable for close-up renders and game assets. | Generate production-ready 3D meshes from text prompts with Tripo P2, featuring native quad topology and mesh budgets up to 50,000 triangles or 25,000 quads. Ideal for game characters and props, hard-surface assets, and high-volume pipelines needing clean, rig-ready geometry. | Generate production-ready 3D models from up to four reference views using Tripo P2.0 Preview, which reconstructs geometry with native quad topology and triangle meshes up to 50,000 faces. Ideal for game characters and props ready for rigging, hard-surface mechanical assets, and high-volume real-time pipelines. | Generate professional 3D assets from 2D images using Tripo engine. | Input a text description of a 3D model. Generate a game-ready 3D model with clean topology and optimized polygon counts | Input a text description or upload reference images. Generate a detailed 3D model with textures, exportable in multiple formats. | Sculpt comprehensive 3D models using Rodin's multi-angle reconstruction. | Transform images or sketches into 3D models with Tripo 3.0's sharp geometry and production-ready PBR textures. | Upload a reference image of your object. Generate a high-detail 3D model with dense geometry and PBR-ready materials. | Upload a reference image of your subject. Generate a game-ready 3D model with clean topology and optimized polygon budgets | Upload an image or multi-view references to generate a clean low-poly 3D model with controlled polygon count and organized topology, ready for export to game engines. | Upload multiview images of your object. Generate a high-detail 3D model with dense geometry and PBR-ready materials. | Upload single or multi-view images to create a 3D model with adjustable quality and enhanced texture | Use an outline reference, and a style reference as input to generate 2D sprite as model textures. Useful for background set dressing in 2D/2.5D game.
- member workflows:
    - api_rodin3d_gen2_5_image_to_3d
    - api_rodin3d_gen2_5_text_to_3d
    - api_rodin_gen2
    - api_rodin_image_to_model
    - api_rodin_multiview_to_model
    - api_tripo3_0_image_to_model
    - api_tripo3_0_text_to_model
    - api_tripo3_1_image_to_model
    - api_tripo3_1_multiview_to_model
    - api_tripo3_1_text_to_model
    - api_tripo_image_to_model
    - api_tripo_multiview_to_model
    - api_tripo_p1_image_to_model
    - api_tripo_p1_mv_to_model
    - api_tripo_p1_text_to_model
    - api_tripo_p2_image_to_model
    - api_tripo_p2_multiview_to_model
    - api_tripo_p2_text_to_model
    - api_tripo_text_to_model
    - templates_3d_match_game_art_style.app
- node clusters (required structure):
    - (none resolved)
- optional roles: Preview3D, LoadImage, MarkdownNote, SaveGLB, SaveImage, BatchImagesNode, GeminiImage2Node, ImageCompare, Note, RecraftRemoveBackgroundNode, Rodin3D_Detail, Rodin3D_Gen2
- unresolved nodes: MarkdownNote, Note

## API / Partner Nodes - 3D / Meshy  (`api_partner_nodes_3d__meshy`)  -  10 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiNanoBanana2, MeshyAnimateModelNode, MeshyImageToModelNode, MeshyImageToModelSmartTopologyNode, MeshyMultiImageToModelNode, MeshyRefineNode, MeshyRigModelNode, MeshyTextToModelNode, MeshyTextToModelSmartTopologyNode, MeshyTextureNode, TripoImageToModelNode)
- when to use: Use to generate a 3D model using Meshy.
- example request: "build a 3d workflow using Meshy"
- description: API image-to-3D via Meshy 6. 1 image -> 1 3D model output. Generates characters, objects, or mechanical parts with production-quality geometry and clean topology. | API multi-image-to-3D via Meshy 6. 3+ images -> 1 3D model output. More input views yield better detail capture, accurate proportions, and cleaner mesh structure. | API text-to-3D via Meshy 6. Text prompt only -> 1 3D model output. Creates characters, mechanical objects, or game-ready low-poly assets with refined geometry. | Generate a 3D model from a single image using Meshy 7.1's Ultra 4K geometry mode, which produces 4096³ meshes so fine surface detail like stitching, wood grain, and scales becomes real geometry rather than texture. Takes one input image. Ideal for high-fidelity asset creation, detailed prop modeling, and product visualization. | Generate a game-ready 3D model from a single image with Meshy T2.5. Meshy T2.5 builds the mesh at the triangle budget you set instead of generating a dense mesh and retopologizing it afterwards, so the result drops straight into a game engine. Texturing with PBR maps is optional and runs at 2k, 4k or 8k. Ideal for turning concept art, product photos and sculpts into engine-ready props. | Generate a game-ready 3D model from a text prompt with Meshy T2.5. Meshy T2.5 builds the mesh at the triangle budget you set instead of generating a dense mesh and retopologizing it afterwards, so the result drops straight into a game engine. No input files required. Ideal for props, environment dressing and low-poly game assets that need a predictable triangle count. | Generate detailed 3D models from text prompts using Meshy 7.1, with an optional Ultra 4K geometry mode that captures fine surface detail like stitching, wood grain, and scales as real geometry. Ideal for game asset creation, product visualization, and 3D printing preparation. | Generate production-ready 3D models from a text prompt with Meshy 7, a text-to-3D model that sets a new standard for geometry alignment, faithfully matching the input's proportions, part placement, and surface details. Ideal for concept art turnarounds, character and prop modeling from sketches, and any workflow where preserving the original design's intent is critical. | Input a single image to automatically generate three orthographic views. Use these multiviews with Meshy to create a textured 3D model. | Turn a single image into a production-ready 3D model with Meshy 7, generating a textured mesh that faithfully mirrors your reference's geometry, proportions, and surface details. Ideal for character busts that preserve facial expression, complex mechanical assemblies with correctly placed parts, and dense surface relief like engraved coins or ornate props.
- member workflows:
    - api_meshy7_1_image_to_model
    - api_meshy7_1_text_to_model
    - api_meshy7_image_to_model
    - api_meshy7_text_to_model
    - api_meshy_image_to_model
    - api_meshy_multi_image_to_model
    - api_meshy_t2_5_smart_topology_i2m
    - api_meshy_t2_5_smart_topology_t2m
    - api_meshy_text_to_model
    - templates_mjm_image_to_3d
- node clusters (required structure):
    - (none resolved)
- optional roles: GeminiNanoBanana2, LoadImage, SaveImage, SaveGLB, BatchImagesNode, MarkdownNote, MeshyAnimateModelNode, MeshyImageToModelNode, MeshyImageToModelSmartTopologyNode, MeshyMultiImageToModelNode, MeshyRefineNode, MeshyRigModelNode
- unresolved nodes: MarkdownNote

## API / Partner Nodes - 3D / Hunyuan3D  (`api_partner_nodes_3d__hunyuan3d`)  -  6 workflow(s)  -  source: official
- execution: api (API nodes: Tencent3DPartNode, TencentImageToModelNode, TencentModelTo3DUVNode, TencentSmartTopologyNode, TencentTextToModelNode)
- when to use: Use to generate a 3D model using Hunyuan3D.
- example request: "build a 3d workflow using Hunyuan3D"
- description: Input a dense mesh (from AI generation or photogrammetry scans) and create a lightweight model with good topology and unwrapped UV. Great for 3D game asset optimization. | Input a text prompt or upload a reference image to generate a detailed 3D model asset. | Upload a 3D model to automatically segment it into its constituent parts, generating a fully decomposed 3D asset for reuse and editing. | Upload a 3D model to perform UV unwrapping. Generate a processed model with optimized UV layout for texturing. | Upload a high-poly 3D model file. Generate a lower-polygon, topologically optimized 3D model with a specified reduction level. | Upload an image to generate a 3D model with geometry and PBR textures.
- member workflows:
    - api_hunyuan3d_image_to_model
    - api_hunyuan3d_model2uv
    - api_hunyuan3d_part
    - api_hunyuan3d_retopo_uv
    - api_hunyuan3d_smart_topology
    - api_hunyuan3d_text_to_model
- node clusters (required structure):
    - other operations: SaveGLB
- optional roles: SaveImage, LoadImage, Load3D, MarkdownNote, Tencent3DPartNode, TencentImageToModelNode, TencentModelTo3DUVNode, TencentSmartTopologyNode, TencentTextToModelNode
- unresolved nodes: MarkdownNote


# API / Partner Nodes - Image to Video  (`api_partner_nodes_image_to_video`)  -  36 workflow(s), 7 model(s)

## API / Partner Nodes - Image to Video / Generic  (`api_partner_nodes_image_to_video__generic`)  -  19 workflow(s)  -  source: official
- execution: api (API nodes: ByteDance2ReferenceNodeV2, ByteDanceImageToVideoNode, GeminiImage2Node, GrokVideoNode, HappyHorseImageToVideoApi, HeyGenImageToVideoNode, LumaRay32ExtendVideoNode, LumaRay32ImageToVideoNode, PixverseImageToVideoNode, PixverseTemplateNode, PixverseV6ImageToVideoNode, PrunaImageToVideoNode, RunwayImageToVideoNodeGen4, Vidu2ImageToVideoNode, Vidu3ImageToVideoNode, Vidu4ImageToVideoNode, ViduImageToVideoNode)
- when to use: Use to generate a video from an input image.
- example request: "build a video workflow"
- description: Animate a static first frame into a short video with synchronized audio using HappyHorse 1.1, taking 1 input image and producing 1 video output. Ideal for e-commerce product demos, brand marketing content, and short episodic series. | Animate a still image with PixVerse V6 Max, keeping the aspect ratio of the input image and generating up to 15 seconds at up to 1080p with an optional native audio track. Ideal for bringing product shots, character art and keyframes to life for social and advertising use. | Generate a 5-15 second clip with synchronized dialogue and sound from a single image using HeyGen Video 1.0, which starts on that exact frame and keeps the image's aspect ratio, at 480p or 768p. The prompt drives both the action and the spoken line, and one video is output. Ideal for giving a portrait or product photo a voice, producing talking-head clips without a shoot, and bringing stills to life. | Generate cinematic 1080P video clips from a single image using Seedance 2.5, animating the still into a moving scene with synchronized sound, music, and dialogue for clips up to 30 seconds across any aspect ratio. Ideal for breathing life into product shots, extending concept art into motion, and creating engaging social media content from photographs. | Generate cinematic videos from a single still image using Vidu Q4 Preview, an expressive flagship model built for lifelike character acting, dramatic camera movement, and striking visual effects. Submit 1 image and receive 1 finished video, with support for up to 15 reference images, 3 reference audio clips, and 2K or 4K output. Ideal for short films, marketing content, and social media storytelling. | Generate cinematic-quality videos from text prompts using Luma Ray 3.2, with support for extending previous generations by providing a generation ID. The workflow accepts 1 input image and produces 1 video output, enabling seamless iterative refinement. Ideal for creating short film sequences, marketing content, and professional video production. | Generate dynamic videos from images using Runway Gen4 Turbo. | Generate dynamic videos from static images with motion and effects using PixVerse. | Generate high-quality, fluid videos from images using HappyHorse-1.0-I2V with realistic dynamic rendering and accurate text and image semantic comprehension. | Generate short videos from a single still image using Grok Imagine Video 1.5 Lite, a faster, lower-cost model that turns one image into a video clip up to 15 seconds at resolutions up to 1080p. Ideal for rapid social content, storyboard previews, and high-volume creative iteration where speed and cost matter more than maximum fidelity. | Transform images into living characters with speech, motion, and synchronized sound effects. | Transform static images into dynamic 1080p videos with cinematic camera control, multi-subject consistency | Transform static images into dynamic 1080p videos with precise motion control and customizable movement amplitude using Vidu. | Transform static images into dynamic videos using ByteDance's Seedance model. Analyzes image structure and generates natural motion with consistent visual style and coherent video sequences. | Turn a single image into video with Pruna P-Video-2. As Pruna's quality-focused video model, its image-to-video mode excels at identity consistency and subtle, natural motion - 720p/1080p, 1-20s duration, plus Draft mode for fast previews. Ideal for image animation, product motion, and poster cinemagraphs. | Turn a still image into a moving video clip with PixVerse V6, generating up to 15 seconds of footage at resolutions from 360p to 1080p. Supports multiple aspect ratios, integrated audio generation, and multi-shot direction for cohesive narrative flow. Ideal for social media ads, product demonstrations, and short-form storytelling. | Upload an image and provide a text prompt to generate a 1-16 second video with synchronized audio and intelligent scene transitions. | Upload your logo, texture and elements. Generate a video of the textured logo for an on brand asset.
- member workflows:
    - api_bytedance_image_to_video
    - api_bytedance_seedance1_5_image_to_video
    - api_grok_imagine_video_1_5_lite_i2v
    - api_happyhorse1_0_i2v
    - api_happyhorse1_1_i2v
    - api_heygen_video_1_i2v
    - api_luma_ray3_3_i2v
    - api_pixverse6_i2v
    - api_pixverse_i2v
    - api_pixverse_template_i2v
    - api_pixverse_v6_max_i2v
    - api_pruna_p_video_i2v
    - api_runway_gen4_turbo_image_to_video
    - api_seedance2_5_i2v_1080p
    - api_vidu_image_to_video
    - api_vidu_q2_i2v
    - api_vidu_q3_image_to_video
    - api_vidu_q4_preview_i2v
    - templates-textured_logo_elements
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
- optional roles: ResizeAndPadImage, BatchImagesNode, ByteDance2ReferenceNodeV2, ByteDanceImageToVideoNode, GeminiImage2Node, GrokVideoNode, HappyHorseImageToVideoApi, HeyGenImageToVideoNode, LumaRay32ExtendVideoNode, LumaRay32ImageToVideoNode, MarkdownNote, PixverseImageToVideoNode
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image to Video / Kling  (`api_partner_nodes_image_to_video__kling`)  -  7 workflow(s)  -  source: mixed
- execution: api (API nodes: ByteDanceSeedreamNodeV2, GeminiImage2Node, GeminiNanoBanana2, GeminiNode, Kling4ImageToVideoNode, KlingImageToVideoWithAudio, KlingOmniProImageToVideoNode, OpenAIChatConfig, OpenAIChatNode)
- when to use: Use to generate a video from an input image using Kling, Seedream.
- example request: "build a video workflow using Kling"
- description: API image-to-video via Kling O3 (Kling 3.0). 1 reference image (+ optional audio/text prompt) -> 1 video output. Generates character-consistent video with native audio output and precise storyboard control. | Animate a single still into a five-second clip with Kling 4.0, cutting three times inside one generation. The source image locks the subject and styling while the default prompt drives the camera moves and a beat-timed edit, and the model also produces stereo audio. Ideal for fashion, editorial, and product motion where the input frame defines the look. | It shows how to go from a simple prompt to a Wes Anderson-style image using a cinematic image model, then enhance it with Nano Banana to improve textures and details, and finally turn it into a video using the current best video model, Kling 3.0. | Transform static images into dynamic videos with Kling O1. Add motion, camera movements, and life to your images using text prompts. | Transform static images into dynamic videos with synchronized dialogue, singing, sound effects, and ambient audio. | Upload an image to automatically generate a descriptive script. This script is then used to create a video with the Kling 3.0 Omni model. | [API] image-to-video via ByteDance Seedream, Kling Omni Pro. images -> video. Tags multiple references with AgentYRefNote and aggregates files for dual API video generation.
- member workflows:
    - akagane-video-batch1
    - api_kling2_6_i2v
    - api_kling4_i2v
    - api_kling_o3_i2v
    - api_kling_omni_i2v
    - template_sirolim_image_script_video
    - templates_ohneis_i2v
- node clusters (required structure):
    - (none resolved)
- optional roles: LoadImage, AgentYRefNote, OpenAIChatConfig, OpenAIChatNode, AgentYImageCollector, SaveImage, bEpicSendToViewer, BatchImagesNode, ByteDanceSeedreamNodeV2, CLIPLoader, CLIPTextEncode, ConditioningZeroOut
- unresolved nodes: MarkdownNote, PrimitiveNode

## API / Partner Nodes - Image to Video / WAN  (`api_partner_nodes_image_to_video__wan`)  -  3 workflow(s)  -  source: official
- execution: api (API nodes: Wan2ImageToVideoApi, Wan3ImageToVideoApi, WanImageToVideoApi)
- when to use: Use to generate a video from an input image using WAN.
- example request: "build a video workflow using WAN"
- description: Create up to 30 seconds of high-fidelity video from a single image using Wan 3.0, which supports text, image, audio, and document inputs for versatile scene generation. Produces authentic, realistic footage with unique facial details and believable motion in one pass. Ideal for cinematic storytelling, product visualization, and documentary-style scene recreation. | Transform images into videos with synchronized audio, enhanced motion, and superior quality. | Upload an image and audio file to generate a video with synchronized sound, using the first and last frames for control.
- member workflows:
    - api_wan2_7_i2v
    - api_wan3_0_i2v
    - api_wan_image_to_video
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
- optional roles: MarkdownNote, LoadAudio, RecordAudio, Wan2ImageToVideoApi, Wan3ImageToVideoApi, WanImageToVideoApi
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image to Video / Flux 3  (`api_partner_nodes_image_to_video__flux_3`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: Flux3ImageToVideoNode, Flux3MaxImageToVideoNode)
- when to use: Use to generate a video from an input image using Flux 3.
- example request: "build a video workflow using Flux 3"
- description: Animate a single still image into a short video clip using FLUX 3 Max, taking one input image and producing one video output. Ideal for bringing product shots, portraits, and concept art to life, generating social media motion content, and rapid prototyping of animated scenes. | Generate full videos from a single text prompt using FLUX 3, which automatically expands your description into a cohesive scene with synchronized audio, motion, and physics. Outputs one 720p, 24fps MP4 video (5-20 seconds) with support for multi-shot structures and aspect ratios from 9:16 to 21:9. Ideal for product photography, filmic ad campaigns, and dynamic motion graphics.
- member workflows:
    - api_bfl_flux3_i2v
    - api_flux3_max_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
- optional roles: Flux3ImageToVideoNode, Flux3MaxImageToVideoNode, MarkdownNote
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image to Video / Gemini  (`api_partner_nodes_image_to_video__gemini`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: GeminiVideoOmniV2)
- when to use: Use to generate a video from an input image using Gemini.
- example request: "build a video workflow using Gemini"
- description: Generate a video from two images using Gemini Omni Flash, which interprets natural language prompts to control duration and aspect ratio. The workflow accepts 2 input images and produces 1 video output. Ideal for creating short brand clips, dynamic social media content, and iterative video edits through conversational prompting. | Generate videos from text prompts or reference images using Gemini Omni 1.1 Flash, a multimodal model that handles text-to-video, image-to-video, keyframe interpolation, and iterative video editing with natural language. Outputs a single MP4 video with optional audio, supporting aspect ratios, resolutions up to 4K, and scene extension. Ideal for cinematic storytelling, product visualization, and rapid video prototyping.
- member workflows:
    - api_google_gemini_omni_flash_1_1_i2v
    - api_google_gemini_omni_flash_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
    - other operations: GeminiVideoOmniV2, MarkdownNote
- optional roles: CustomCombo
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Image to Video / LTX-2  (`api_partner_nodes_image_to_video__ltx_2`)  -  2 workflow(s)  -  source: mixed
- execution: api (API nodes: LtxApi25ImageToVideo)
- when to use: Use to generate a video from an input image using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate video from a single image using LTX-2.5's fast distilled model, delivering cinematic multi-shot sequences with consistent characters, lighting, and style across cuts. The workflow outputs a video file with native multishot support and automatic duration prediction. Ideal for rapid storyboarding, social media content creation, and pre-visualization in commercial production pipelines.
- member workflows:
    - api_ltx2_5_i2v
    - api_ltxv_image_to_video
- node clusters (required structure):
    - inputs: LoadImage
- optional roles: GetVideoComponents, LtxApi25ImageToVideo, LtxvApiImageToVideo, SaveVideo, VHS_VideoCombine
- unresolved nodes: LtxvApiImageToVideo

## API / Partner Nodes - Image to Video / WAN 2.6  (`api_partner_nodes_image_to_video__wan_2_6`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: WanImageToVideoApi)
- when to use: Use to generate a video from an input image using WAN 2.6.
- example request: "build a video workflow using WAN 2.6"
- description: API image-to-video via Wan 2.6. 1 image -> 1 video output. Generates 1080P video with enhanced image quality, smoother motion, and natural movement.
- member workflows:
    - api_wan2_6_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: VHS_VideoCombine
    - other operations: GetVideoComponents, WanImageToVideoApi


# Video to Video  (`video_to_video`)  -  31 workflow(s), 10 model(s)

## Video to Video / LTX-2  (`video_to_video__ltx_2`)  -  7 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video guided by a control map (canny/depth/pose) using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate LTX 2.3 videos with IC-LoRA using aligned control inputs like depth, pose, or edges. | Generate high-quality videos from depth maps with synchronized audio-video generation using LTX-2. Features precise depth control, spatial awareness, and efficient performance. | Generate high-quality videos from edge detection (Canny) guidance with synchronized audio-video generation using LTX-2. Features precise edge control, structural consistency, and efficient performance. | Generates depth-controlled video with LTX-2: motion and structure follow a depth-reference video alongside text prompting, optional first-frame image conditioning, with optional synchronized audio. | Generates video from Canny edge maps using LTX-2, with optional synchronized audio. | Generates video from pose reference frames using LTX-2, with optional synchronized audio. | [Local] video-to-video via LTX-2 19b with pose-control and distilled LoRAs. Video plus reference images -> video. Use for restyling or pose-guided edits of an existing clip through LTX-2.
- member workflows:
    - bepic_templates__bEpic_Vid2VId_LTX2
    - canny_to_video_ltx_2_0
    - depth_to_video_ltx_2_0
    - pose_to_video_ltx_2_0
    - video_ltx2_3_ic_lora
    - video_ltx2_canny_to_video
    - video_ltx2_depth_to_video
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader, LoraLoaderModelOnly
    - conditioning: CLIPTextEncode (x2), LTXVConditioning
    - latent / canvas: LTXVEmptyLatentAudio
    - other operations: EmptyLTXVLatentVideo, LTXVAddGuide, LTXVConcatAVLatent, LTXVCropGuides, LTXVImgToVideoInplace, LTXVSeparateAVLatent
- paired/multiple required: CLIPTextEncode x2
- optional roles: GetNode, SetNode, KSamplerSelect, SamplerCustomAdvanced, bEpicReformat, CFGGuider, RandomNoise, ManualSigmas, MarkdownNote, MoGeRender, String, VAEDecode
- unresolved nodes: GetNode, Image Load, MarkdownNote, Reroute, SetNode, String, easy cleanGpuUsed

## Video to Video / WAN VACE  (`video_to_video__wan_vace`)  -  6 workflow(s)  -  source: custom
- execution: local
- when to use: Use to edit an existing video using WAN VACE, WAN 2.2.
- example request: "build a video workflow using WAN VACE"
- description: [Local] image editing via Wan. 3 image inputs -> 1 image output. Performs advanced image-to-image editing and transformations. | [Local] ldr-to-hdr reconstruction via Wan2.1-VACE DiffHDR. video -> image. Reconstructs linear hdr from ldr footage using sliding window blending. | [Local] video-to-video via Wan VACE and Canny preprocessor. Video plus reference images -> video. ControlNet-style VACE re-render of a source clip with keyframe replacement.
- member workflows:
    - ComfyUI-DiffHDR__diffhdr_video
    - Wan22Vace_VID2VID
    - bepic_templates__bEpic_VID2VID_WanVace
    - video_wan_vace_14B_ref2v
    - video_wan_vace_14B_v2v
    - video_wan_vace_outpainting
- node clusters (required structure):
    - model loading: VAELoader
- optional roles: GetNode, SetNode, bEpicReformat, String, bEpicSendToViewer, AIO_Preprocessor, CLIPTextEncode, DiffusionModelLoaderKJ, DiffusionModelSelector, GetVideoComponents, Image Load, KSamplerAdvanced
- unresolved nodes: GetNode, Image Load, Int, SetNode, String, easy cleanGpuUsed, easy mathInt

## Video to Video / WAN  (`video_to_video__wan`)  -  5 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video using WAN, SCAIL.
- example request: "build a video workflow using WAN"
- description: Animate a reference character using a driving video or replace characters entirely with SCAIL-2 Int8, which delivers higher quality than fp8 while running faster. The workflow accepts one reference image and one driving video, outputting an animated character sequence with replaced identity. Ideal for cross-identity character swaps, animal-driven animation, and multi-character scenes without intermediate pose representations. | Animate a still character using a driving video with Wan Animate 2, transferring motion directly from video frames without any pose extraction or skeleton preprocessing. The output preserves the reference character's identity while generating a fresh background and camera movement from your text prompt. Ideal for motion transfer, action mimicry, and character animation replacement projects. | Animate a still character using a driving video with Wan Animate 2, transferring motion directly from video frames without any pose extraction or skeleton preprocessing. The output preserves the reference character's identity while generating a fresh background and camera movement from your text prompt. This distilled version delivers efficient, high-fidelity motion generation with strong identity preservation, ideal for motion transfer, action mimicry, and character animation replacement projects. | Generate videos guided by pose, depth, and edge controls using Wan 2.1 ControlNet. | Upload 1 reference character image and 1 driving video. Generate an animated video of the character matching the driving video's motion. Ideal for character animation, in-video character replacement, and motion transfer projects.
- member workflows:
    - video_wan21_scail2_character_replacement
    - video_wan21_scail2_character_replacement_int8
    - video_wan_animate2
    - video_wan_animate2_distilled
    - wan2.1_fun_control
- node clusters (required structure):
    - inputs: LoadImage, LoadVideo
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CLIPVisionEncode, CLIPVisionLoader, CreateVideo, GetVideoComponents, MarkdownNote, ModelSamplingSD3
- paired/multiple required: CLIPTextEncode x2
- optional roles: ImageFromBatch, ResizeImageMaskNode, LoraLoaderModelOnly, PreviewImage, SAM3_VideoTrack, BasicScheduler, CheckpointLoaderSimple, ContextWindowsManual, GetItemFromList, KSamplerSelect, SCAIL2ColoredMask, SamplerCustom
- unresolved nodes: MarkdownNote

## Video to Video / Generic  (`video_to_video__generic`)  -  3 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video.
- example request: "build a video workflow"
- description: Upload a video and select either RIFE or FILM frame interpolation model to generate smoother, higher frame rate video output. | [Local] Video decomposition via UniVidX nodes. 1 video -> 4 images. Use this to extract intrinsic and alpha layers from video clips. | [Local] video decomposition via UniVidX. video -> four images. Splits scene materials, lighting, and alpha channels for compositing.
- member workflows:
    - UniVidX_ComfyUI__R2AIN_video_api
    - UniVidX_ComfyUI__R2PFB_video_api
    - utility_video_frame_interpolation
- node clusters (required structure):
    - (none resolved)
- optional roles: SaveImage, CreateVideo, FrameInterpolate, FrameInterpolationModelLoader, GetVideoComponents, LoadVideo, MarkdownNote, SaveVideo, UniVidXDecodeAlpha, UniVidXDecodeIntrinsic, UniVidXLoader, UniVidXSampler
- unresolved nodes: MarkdownNote

## Video to Video / WAN 2.2  (`video_to_video__wan_2_2`)  -  3 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video guided by a control map (canny/depth/pose) using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Multi-condition video control with pose, depth, and edge guidance. Compact 5B size for experimental development. | [Local] video-to-video animation via Wan 2.2 Animate 14B and WanAnimateToVideo. Input video plus reference images -> animated video. Pick for driving a subject video with an animate model rather than plain text-to-video.
- member workflows:
    - bepic_templates__bEpic_VID2VID_WanAnimate
    - video_wan2_2_14B_fun_control
    - video_wan2_2_5B_fun_control
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - other operations: ModelSamplingSD3
- paired/multiple required: CLIPTextEncode x2
- optional roles: GetNode, SetNode, bEpicReformat, String, bEpicSendToViewer, DWPreprocessor, GetVideoComponents, Image Load, KSamplerAdvanced, LoraLoaderModelOnly, MarkdownNote, PreviewImage
- unresolved nodes: GetNode, Image Load, MarkdownNote, SetNode, String, easy cleanGpuUsed

## Video to Video / SAM3  (`video_to_video__sam3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to segment an image using SAM3.
- example request: "build an image workflow using SAM3"
- description: Extract a full-body 3D human mesh from a single video frame using SAM 3D Body, guided by a text prompt to refine pose and shape estimation. The workflow loads one video and one text prompt, then outputs a reconstructed video showing the recovered human rig. Ideal for motion capture prototyping, biomechanics analysis, and character animation previsualization. | Use the SAM3 model to segment the main subject or content from a video, isolating specific objects or regions. Input a video and receive a segmented mask output.
- member workflows:
    - utility_sam3d_body
    - utility_video_segment_sam3
- node clusters (required structure):
    - inputs: LoadVideo
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode
    - other operations: GetVideoComponents, MarkdownNote, Note
- optional roles: BuildPoseFile, CreateVideo, LoadMoGeModel, MaskPreview, MoGeGeometryToFOV, MoGeInference, RTDETR_detect, SAM3DBody_FaceExpression, SAM3DBody_Loader, SAM3DBody_Predict, SAM3DBody_Render, SAM3DBody_Smooth
- unresolved nodes: MarkdownNote, Note

## Video to Video / SDPose  (`video_to_video__sdpose`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a pose map using SDPose.
- example request: "build a video workflow using SDPose"
- description: Upload a video to detect human poses. Supports detection for both single individuals and multiple people within the same scene | Upload a video to extract pose keypoints and generate a pose map. The workflow supports multiple person detection and uses an enhanced SDPose model for accurate whole-body feature extraction.
- member workflows:
    - utility_sdpose_multi_person_video
    - utility_sdpose_ood_video_to_pose_map
- node clusters (required structure):
    - inputs: LoadVideo
    - model loading: CheckpointLoaderSimple
    - output: SaveVideo
    - other operations: CreateVideo, GetVideoComponents, MarkdownNote, ResizeImageMaskNode, SDPoseDrawKeypoints, SDPoseKeypointExtractor
- optional roles: DrawBBoxes, ImageBlend, PreviewImage, RTDETR_detect, UNETLoader
- unresolved nodes: MarkdownNote

## Video to Video / Bernini  (`video_to_video__bernini`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing video using Bernini.
- example request: "build a video workflow using Bernini"
- description: Generate an edited video with consistent relighting using Bernini-R. Ideal for portrait and product relighting, e-commerce catalog photography, and creating consistent lighting across photo sets.
- member workflows:
    - video_bernini_r_video_editing
- node clusters (required structure):
    - inputs: LoadImage, LoadVideo
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: BatchImagesNode, ComfySwitchNode (x5), CreateVideo, CustomCombo, GetImageSize, GetVideoComponents (x2), MarkdownNote (x4), PreviewAny (x2), PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, StringConcatenate, StringReplace, Video Slice
- paired/multiple required: MarkdownNote x4, CLIPTextEncode x2, GetVideoComponents x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: MarkdownNote

## Video to Video / Depth Anything  (`video_to_video__depth_anything`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to edit an existing video using Bernini, Depth Anything.
- example request: "build a video workflow using Bernini"
- description: This subgraph uses Depth Anything 3 to predict spatially consistent geometry from any number of images or video frames, with or without known camera poses. It outputs depth maps, camera poses, and optionally 3D Gaussian parameters for novel view synthesis.
- member workflows:
    - video_edit_bernini_r
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler, BerniniConditioning, CLIPTextEncode (x2), SplitSigmas
    - sampling: KSamplerSelect, SamplerCustom (x2)
    - decoding: VAEDecode
    - other operations: ComfySwitchNode (x5), CreateVideo, CustomCombo, GetVideoComponents, MarkdownNote, PreviewAny, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x5), PrimitiveStringMultiline, RegexExtract, StringConcatenate, StringReplace
- paired/multiple required: CLIPTextEncode x2, LoraLoaderModelOnly x2, SamplerCustom x2, UNETLoader x2
- unresolved nodes: MarkdownNote

## Video to Video / MiniMax  (`video_to_video__minimax`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video guided by a control map (canny/depth/pose) using MiniMax.
- example request: "build a video workflow using MiniMax"
- description: Generate videos from a pose reference video using MiniMax-H3 with a ControlNet-Union checkpoint, preserving the input's motion and structure while applying your prompt. Ideal for character animation, motion transfer, and choreography visualization projects. This workflow is for pose control only
- member workflows:
    - video_minimax_h3_fun_controlnet_union
- node clusters (required structure):
    - inputs: LoadVideo
    - model loading: CLIPLoader, CheckpointLoaderSimple, LoraLoaderModelOnly, ModelPatchLoader, UNETLoader (x2), VAELoader (x2)
    - conditioning: BasicGuider, BasicScheduler, MiniMaxH3FunControlNetApply
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode, VAEDecodeAudio
    - output: PreviewImage, SaveVideo
    - other operations: ComfyMathExpression, ComfySwitchNode (x2), CreateVideo, GetImageSize, GetVideoComponents (x2), MarkdownNote (x6), MiniMaxH3ReferenceToVideo, PrimitiveBoolean (x2), PrimitiveFloat, PrimitiveInt (x2), PrimitiveStringMultiline, RTDETR_detect, RandomNoise, ResizeImageMaskNode, ResolutionSelector, SDPoseDrawKeypoints, SDPoseKeypointExtractor, Video Slice
- paired/multiple required: MarkdownNote x6, GetVideoComponents x2, UNETLoader x2, VAELoader x2
- unresolved nodes: MarkdownNote


# API / Partner Nodes - Audio  (`api_partner_nodes_audio`)  -  26 workflow(s), 3 model(s)

## API / Partner Nodes - Audio / Generic  (`api_partner_nodes_audio__generic`)  -  24 workflow(s)  -  source: official
- execution: api (API nodes: ByteDanceSeedAudio, ElevenLabsAudioIsolation, ElevenLabsInstantVoiceClone, ElevenLabsSpeechToSpeech, ElevenLabsTextToDialogue, ElevenLabsTextToSoundEffects, ElevenLabsTextToSpeech, ElevenLabsVoiceSelector, FishAudioInstantVoiceClone, FishAudioSpeechToText, FishAudioTextToSpeech, FishAudioVoiceSelector, HeyGenTextToSpeechNode, HeyGenVoiceCloneNode, HeyGenVoiceTextToSpeechNode, MireloTextToSFX, MireloVideoToSFX, SoniloTextToMusic, SoniloVideoToMusic, VeedCleanAudioNode, VeedCleanVideoAudioNode)
- when to use: Use to generate audio from a text prompt.
- example request: "build an audio workflow"
- description: Add picture-locked sound effects to a clip with Mirelo sfx-2.0: the model watches the video and times Foley to on-screen action, keeps speech by default, and saves the SFX track, the speech track and the video carrying the new mix. | Clean the speech inside a video with VEED Clean Audio: background noise, music and effects are stripped from the soundtrack, pauses are cleaned completely, and the clip comes back with 48 kHz mono speech, optionally normalized to a target loudness while the picture stays as shot. Video up to 1799 seconds. | Clone a voice from a short recording with HeyGen Voice 1.0 Turbo, then speak any script in that voice. It is the same instant clone as Voice 1.0, run through the turbo model, which renders faster for quick drafts and iteration. Connect one audio clip to Instant Clone and the resulting voice_id feeds straight into text to speech, while SaveText keeps the ID on disk for the next session. Ideal for narration in your own voice, character dialogue, and multilingual voiceover. | Clone a voice from a short recording with HeyGen Voice 1.0, then speak any script in that voice. Connect one audio clip to Instant Clone and the resulting voice_id feeds straight into text to speech, with no training wait. Use the expressiveness control to stay close to the recording or push more emotion into the delivery. Ideal for narration in your own voice, character dialogue, and multilingual voiceover. | Clone any voice from a short recording, then use it to speak any script with Fish Audio. Record 2-3 clips of 15-20 seconds, connect them to the Instant Voice Clone node, and the cloned voice feeds straight into text to speech with no training wait. Add emotion tags to control how your cloned voice delivers each line. Great for personalized narration, character voices, and content in your own voice. | Direct a single Fish Audio Drama 3 performance from the text: shift the delivery mid-line with [plain-language directions] and wrap words with <whisper>...</whisper> tags. | Generate a multi-speaker dialogue from text with Eleven v4: give every entry its own line and its own voice, direct the delivery with audio tags such as [quietly] or [laughs], or clone a voice from a sample and let it take a turn. One audio file is output. Ideal for podcasts, game dialogue, and character conversations.
 | Generate a multi-speaker scene in one pass with Fish Audio Drama 3: connect one Voice Selector per speaker, mark speaker changes with @Voice1 / @Voice2, and direct the delivery with plain-language [directions] and <whisper> word tags. | Generate expressive speech from text with Eleven v4, the quality-focused ElevenLabs model. Direct the performance with audio tags such as [whispers] or [excited] written into the script, or upload a voice sample to clone a voice and narrate the script in it. One audio file is output. Ideal for audiobooks, narration, and character voiceovers. | Generate high-quality, production-ready music from text prompts. Input descriptive text to create original soundtracks with streaming playback and precise duration control. | Generate natural, expressive speech from text with Fish Audio S2 Pro, a text-to-speech model supporting 80+ languages and fine-grained control over emotion, tone, and pacing using inline tags like [whisper] or [excited]. Input your text and receive one high-quality audio output, with optional style tags woven directly into the script. Ideal for voiceover production, multilingual content creation, and character dialogue in games or animations. | Generate natural-sounding speech from text using HeyGen's advanced TTS engine, supporting a wide range of languages and accents with customizable voices. Input your text and select from numerous preset voices or clone a custom voice from a short audio sample. Ideal for creating voiceovers for videos, generating multilingual narration, and producing personalized audio content. | Generate sound effects from a text prompt with Mirelo sfx-2.0: name the sound, its material, space and intensity in one line and get a 48 kHz mono audio file, up to 10 minutes. | Generate synchronized soundtracks from video footage. Input a video to produce music that matches its pacing and emotional cues, outputting a perfectly timed audio file. | Input a text prompt to generate custom sound effects and ambient audio. | Input a text prompt to generate speech, dialogue, background music, and sound effects in one audio file. Describe voices, emotion, and scene details to create multi-speaker audio up to 2 minutes. | Input text and select a voice profile to generate a high-quality, emotionally expressive audio dialogue. | Input text to generate speech with ultra-realistic voices, or upload a voice sample to clone it for synthesis. | Remove background noise from a speech recording with VEED Clean Audio: music and effects go with the noise, pauses are cleaned completely, and the voice comes back as 48 kHz mono audio, optionally normalized to a target loudness. Audio up to 1799 seconds. | Transcribe audio to text with Fish Audio ASR. Language is detected automatically across 80+ languages, with optional word-level timestamps returned as JSON segments. Use it to create subtitles, transcribe recordings, or generate text prompts from audio clips. A sample audio is pre-loaded so you can test it right away. | Upload a character image and write a text prompt to generate audio. The model derives the voice from your image, then produces speech, ambience, background music, and sound effects. | Upload a reference audio clip and write a text prompt to clone the voice into a new scene. Generate dialogue, background music, and sound effects with up to 3 reference voices. | Upload a source audio file to apply a new voice tone or clone a different voice, generating a modified audio output. | Upload an audio file containing background noise. Use the ElevenLabs API to isolate and output a clean voice track.
- member workflows:
    - api_bytedance_seed_audio1_0_t2a
    - api_bytedance_seed_audio1_0_ta2a
    - api_bytedance_seed_audio1_0_ti2a
    - api_elevenlabs_speech_to_speech
    - api_elevenlabs_text_to_dialogue
    - api_elevenlabs_text_to_sound_effects
    - api_elevenlabs_text_to_speech
    - api_elevenlabs_v4_text_to_dialogue
    - api_elevenlabs_v4_text_to_speech
    - api_elevenlabs_voice_isolation
    - api_fishaudio_drama3_dialogue
    - api_fishaudio_drama3_directed
    - api_fishaudio_speech_to_text
    - api_fishaudio_text_to_speech
    - api_fishaudio_voice_clone_tts
    - api_heygen_text_to_speech
    - api_heygen_voice_1
    - api_heygen_voice_1_turbo
    - api_mirelo_text_to_sfx
    - api_mirelo_video_to_sfx
    - api_sonilo_t2m
    - api_sonilo_v2m
    - api_veed_clean_audio
    - api_veed_clean_audio_video
- node clusters (required structure):
    - (none resolved)
- optional roles: AudioAdjustVolume, ElevenLabsVoiceSelector, FishAudioVoiceSelector, LoadAudio, MarkdownNote, SaveAudioAdvanced, SaveText, AudioMerge, ByteDanceSeedAudio, CreateVideo, ElevenLabsAudioIsolation, ElevenLabsInstantVoiceClone
- unresolved nodes: MarkdownNote, Note

## API / Partner Nodes - Audio / ACE-Step  (`api_partner_nodes_audio__ace_step`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: GeminiNodeV2)
- when to use: Use to generate audio from a text prompt using ACE-Step.
- example request: "build an audio workflow using ACE-Step"
- description: Input a music style and optional lyrics. Generate a full song with AI-created lyrics and audio.
- member workflows:
    - audio_ace_step_1_5_split_llm
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: ConditioningZeroOut, TextEncodeAceStepAudio1.5
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - output: SaveAudioAdvanced
    - other operations: EmptyAceStep1.5LatentAudio, GeminiNodeV2, MarkdownNote, ModelSamplingAuraFlow, PreviewAny, PrimitiveFloat, RegexExtract (x2), SeedNode
- paired/multiple required: RegexExtract x2
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Audio / LTX-2  (`api_partner_nodes_audio__ltx_2`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: ElevenLabsInstantVoiceClone, ElevenLabsTextToSpeech, ElevenLabsVoiceSelector, GeminiNode)
- when to use: Use to generate a video using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Upload an image to generate AI prompts and synthesize speech. Create a final video with accurate lip sync using ElevenLabs and LTX Video.
- member workflows:
    - template_image_speech_to_video
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple, LTXVAudioVAELoader, LatentUpscaleModelLoader, LoraLoader, LoraLoaderModelOnly (x2)
    - conditioning: CFGGuider (x2), CLIPTextEncode (x2), LTXAVTextEncoderLoader, LTXVConditioning, ManualSigmas (x2)
    - latent / canvas: LTXVAudioVAEEncode
    - sampling: KSamplerSelect (x2), LTXVLatentUpsampler, SamplerCustomAdvanced (x2)
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - output: SaveAudioMP3, SaveVideo
    - other operations: ComfyMathExpression (x4), CreateVideo, ElevenLabsInstantVoiceClone, ElevenLabsTextToSpeech, ElevenLabsVoiceSelector, EmptyLTXVLatentVideo, GeminiNode, LTXVConcatAVLatent (x2), LTXVCropGuides, LTXVImgToVideoInplace (x2), LTXVPreprocess, LTXVSeparateAVLatent (x2), LoadAudio, MarkdownNote (x3), PreviewAny (x4), PrimitiveBoolean, PrimitiveFloat, PrimitiveInt (x3), PrimitiveStringMultiline, RandomNoise (x2), RegexExtract (x2), Reroute, ResizeImageMaskNode, ResizeImagesByLongerEdge, SetLatentNoiseMask, SolidMask, TextGenerateLTX2Prompt, TrimAudioDuration
- paired/multiple required: MarkdownNote x3, CFGGuider x2, CLIPTextEncode x2, KSamplerSelect x2, LTXVConcatAVLatent x2, LTXVImgToVideoInplace x2, LTXVSeparateAVLatent x2, LoraLoaderModelOnly x2, ManualSigmas x2, RandomNoise x2, RegexExtract x2, SamplerCustomAdvanced x2
- unresolved nodes: MarkdownNote, Reroute


# API / Partner Nodes - First / Last Frame to Video  (`api_partner_nodes_first_last_frame_to_video`)  -  21 workflow(s), 5 model(s)

## API / Partner Nodes - First / Last Frame to Video / Generic  (`api_partner_nodes_first_last_frame_to_video__generic`)  -  12 workflow(s)  -  source: mixed
- execution: api (API nodes: ByteDance2DraftToFinalVideoNode, ByteDance2FirstLastFrameNode, ByteDanceCreateImageAsset, ByteDanceFirstLastFrameNode, ByteDanceSeedreamNodeV3, GeminiImage2Node, GeminiImageNode, GeminiNanoBanana2V2, GeminiNode, GeminiNodeV3, KlingFirstLastFrameNode, Vidu2StartEndToVideoNode)
- when to use: Use to generate a video interpolating between a first and last frame.
- example request: "build a video workflow"
- description: API first-last-frame-to-video via Seedance 2.0 (ByteDance). 1 first frame image + 1 optional last frame image -> 1 video output. Generates video interpolated between keyframes with precise motion control. | Create smooth video transitions between start and end frames with cinematic camera movements, multi-subject consistency. | Generate a smooth video transition between a user-provided start frame and end frame with Seedance 2.5, which automatically interpolates the motion, camera movement, and visual evolution while locking both endpoints. Ideal for shot transitions, storyboard previz, and before-and-after scene showcases like empty-to-packed or day-to-night shifts. | Generate cinematic video transitions between start and end frames with fluid motion, scene consistency, and professional polish using ByteDance's Seedance model. | Generate draft preview videos from a first frame image using Seedance 2.5's two-phase draft mode: stage one turns your image into a fast, low-cost 480p preview plus a reusable draft task ID, and stage two renders the 1080p final from that ID. Ideal for testing motion, camera and scene ideas on a still before committing to a full-quality render. | Upload a photo of your vehicle and generate a studio quality video of the vehicle from multiple angles. | Upload a reference image of a real person for identity verification and generate a personalized video with first and last frame control using Seedance2.0 FLF2V. | Upload a start frame and end frame, and Seedance 1.5 Pro automatically generates a smooth video transition with synchronized audio between them. | Upload a vector image of your logo, and prompt your desired texture. Generate a textured 3D first and last frame with automated prompting for the final animation. | Upload an image of a device or object to generate start and end frames. The workflow animates the transition to an exploded view and back, outputting a seamless video loop. | Upload the first and last frames of your video. Generate a high-quality video sequence using the Seedance2.0 model. | Upload your logotype and apply a texture + elements for on brand asset
- member workflows:
    - api_bytedance_flf2v
    - api_bytedance_seedance1_5_flf2v
    - api_seedance2_0_flf2v
    - api_seedance2_0_flf2v_real_human
    - api_seedance2_5_draft_i2v
    - api_seedance2_5_flf2v
    - api_seedance2_i2v_flf
    - api_vidu_q2_flf2v
    - template_eric_exploded_view
    - templates-3D_logo_texture_animation
    - templates-car_product
    - templates-textured_logotype-v2.1
- node clusters (required structure):
    - inputs: LoadImage
- optional roles: RegexExtract, ByteDanceSeedreamNodeV3, SaveImage, SaveVideo, GeminiImage2Node, GeminiNodeV3, GetVideoComponents, KlingFirstLastFrameNode, MarkdownNote, ByteDanceCreateImageAsset, ByteDanceFirstLastFrameNode, GeminiNanoBanana2V2
- unresolved nodes: MarkdownNote, Note, Reroute

## API / Partner Nodes - First / Last Frame to Video / MiniMax  (`api_partner_nodes_first_last_frame_to_video__minimax`)  -  4 workflow(s)  -  source: official
- execution: api (API nodes: MinimaxHailuo03ContextIRNode, MinimaxHailuo03FirstLastFrameNode, MinimaxHailuo03RegenerateNode)
- when to use: Use to generate a video from an input image using MiniMax.
- example request: "build a video workflow using MiniMax"
- description: Generate a video from a first and last frame using MiniMax H3, producing a smooth animation between the two images. You provide 2 images, and the workflow outputs a single video clip. Ideal for creating seamless transitions in animation, visualizing before-and-after transformations, and generating consistent motion for character or scene sequences. | Generate polished, prompt-faithful videos faster than real time with MiniMax H3 Max, a post-trained model tuned for stronger prompt adherence and vivid cinematic aesthetics. This workflow supports text-to-video, image-to-video animation, and directed first-to-last-frame transitions, delivering clean compositions and cohesive lighting directly from the model. Ideal for rapid concept exploration, cinematic motion graphics, and high-throughput video comparison. | Generate video from a source image using MiniMax H3 Max, animating your still frame while preserving its composition, visual identity, and aspect ratio. The model delivers faster-than-real-time generation with strong prompt adherence and polished cinematic aesthetics. Ideal for bringing character art to life, animating product shots, and creating motion from concept images. | Generate video sequences from a starting image using MiniMax H3 Max Turbo, producing two video outputs from a single input frame. This rapid image-to-video workflow generates short clips in seconds, with optional prompt enhancement and 2K upscaling stages. Ideal for animating still photographs, creating quick motion previews for storyboards, and fast visual effects iteration.
- member workflows:
    - api_minimax_h3_flf2v
    - api_minimax_h3_max_flf2v
    - api_minimax_h3_max_i2v
    - api_minimax_h3_max_turbo_i2v
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveVideo
    - other operations: MinimaxHailuo03FirstLastFrameNode
- optional roles: MarkdownNote, MinimaxHailuo03ContextIRNode, MinimaxHailuo03RegenerateNode, SaveText
- unresolved nodes: MarkdownNote

## API / Partner Nodes - First / Last Frame to Video / Kling  (`api_partner_nodes_first_last_frame_to_video__kling`)  -  3 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiNode, KlingFirstLastFrameNode, KlingOmniProFirstLastFrameNode)
- when to use: Use to generate a video interpolating between a first and last frame using Kling.
- example request: "build a video workflow using Kling"
- description: API first-last-frame-to-video via Kling O3 (Kling 3.0). Up to 4 reference/keyframe images -> 1 video output. Generates videos with precise semantic control, longer duration, and improved narrative coherence. | Input a first and last frame image to generate a continuous video sequence with multi-shot generation, precise element control, and support for multilingual prompts. | Upload 8 images (in order) to be used as keyframes for 7 Kling 3.0 FLF2V videos.
- member workflows:
    - api_kling_o3_flf2v
    - api_kling_v3_flf2v
    - template_contact_sheet-step_3.app
- node clusters (required structure):
    - inputs: LoadImage (x2)
- paired/multiple required: LoadImage x2
- optional roles: ImageFromBatch, BatchImagesNode, GeminiNode, KlingFirstLastFrameNode, SaveVideo, GetVideoComponents, ImageBatchMulti, KlingOmniProFirstLastFrameNode, VHS_VideoCombine
- unresolved nodes: PrimitiveNode, Reroute

## API / Partner Nodes - First / Last Frame to Video / LTX-2  (`api_partner_nodes_first_last_frame_to_video__ltx_2`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: LtxApi25ImageToVideo)
- when to use: Use to generate a video interpolating between a first and last frame using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate a video from two reference images using LTX-2.5, blending the content of both frames into a single animated sequence. Ideal for creating transition shots, morphing between two subjects or scenes, and building multi-shot narratives from still references.
- member workflows:
    - api_ltx2_5_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - output: SaveVideo
    - other operations: LtxApi25ImageToVideo
- paired/multiple required: LoadImage x2

## API / Partner Nodes - First / Last Frame to Video / WAN 2.2  (`api_partner_nodes_first_last_frame_to_video__wan_2_2`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: GeminiImageNode, GeminiNodeV3)
- when to use: Use to generate a video using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Upload a starting image and its prompt to generate a looping video from the first to the last frame.
- member workflows:
    - templates_mjm_looped_restyler
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader (x2), LoraLoaderModelOnly (x4), UNETLoader (x4), VAELoader (x2)
    - conditioning: CLIPTextEncode (x4)
    - sampling: KSamplerAdvanced (x4)
    - decoding: VAEDecode (x2)
    - output: PreviewImage (x3), SaveVideo
    - other operations: BatchImagesNode, CreateVideo (x3), GeminiImageNode, GeminiNodeV3 (x2), GetImageSize, GetVideoComponents (x2), ImageScaleToTotalPixels (x2), ImageStitch, MarkdownNote, ModelSamplingSD3 (x4), SplitImageWithAlpha, WanFirstLastFrameToVideo (x2)
- paired/multiple required: CLIPTextEncode x4, KSamplerAdvanced x4, LoraLoaderModelOnly x4, ModelSamplingSD3 x4, UNETLoader x4, CreateVideo x3, PreviewImage x3, CLIPLoader x2, GeminiNodeV3 x2, GetVideoComponents x2, ImageScaleToTotalPixels x2, VAEDecode x2, VAELoader x2, WanFirstLastFrameToVideo x2
- unresolved nodes: MarkdownNote


# API / Partner Nodes - Upscale  (`api_partner_nodes_upscale`)  -  21 workflow(s), 5 model(s)

## API / Partner Nodes - Upscale / Generic  (`api_partner_nodes_upscale__generic`)  -  9 workflow(s)  -  source: official
- execution: api (API nodes: ByteDanceVideoEnhanceNode, ClarityCrystalUpscaleNode, ClarityCrystalVideoUpscaleNode, HitPawGeneralImageEnhance, HitPawVideoEnhance, RecraftCreativeUpscaleNode, RecraftCrispUpscaleNode, WavespeedFlashVSRNode, WavespeedImageUpscaleNode)
- when to use: Use to upscale / enhance an image.
- example request: "build an image workflow"
- description: Enhance video quality with ByteDance vCube, an AI-driven tool that upscales and restores footage through shot-adaptive processing, fixing compression artifacts and color distortion while preserving original texture. Ideal for improving UGC clips, refining AIGC content, and restoring classic films with vibrant, clear results. | Upload a low-quality portrait image and adjust the upscaling factor. Generate a high-fidelity, detailed portrait with enhanced skin textures and facial features. | Upload a low-resolution or noisy image to restore and upscale. Generate a high-resolution output with enhanced details, reduced artifacts, and preserved natural look. | Upload a video to enhance and upscale. Generate a restored output with improved clarity, temporal stability, and optional portrait enhancement. | Upload a video to upscale it to 720p, 1080p, 2K, or 4K. The workflow enhances detail, reduces artifacts, and ensures consistent motion for a high-quality, natural-looking output. | Upload an image to enhance and upscale it. Recraft's AI reconstructs details and improves textures, outputting a higher-quality image with a natural appearance. | Upload an image to upscale its resolution. Generate an enhanced 2K, 4K, or 8K output that preserves details and reduces artifacts in JPEG, PNG, or WEBP format. | Upscale a video with Clarity AI's Crystal model, running through the Fal proxy: one video in, one upscaled video out. Set scale_factor to multiply the frame size, and the output keeps the source resolution, frame count and audio track. The result is capped at 9 megapixels per frame and 360 megapixel-seconds per clip, so long or oversized clips come back upscaled by less than the factor you asked for. Ideal for restoring generated clips, preparing footage for large screens, and sharpening soft AI video. | Upscale and enhance images with the precision-focused Crystal model, preserving real detail while adding resolution. Takes one image in and returns one upscaled image plus a side-by-side comparison view. Ideal for portraits, product shots, and print-ready enlargements where faithful detail matters.
- member workflows:
    - api_bytedance_vcube_video_enhance
    - api_clarity_ai_crystal_upscale
    - api_clarity_ai_crystal_video_upscale
    - api_wavespeed_flshvsr_video_upscale
    - api_wavespeed_image_upscale
    - utility_hitpaw_general_image_enhance
    - utility_hitpaw_video_enhance
    - utility_recraft_creative_image_upscale
    - utility_recraft_crisp_image_upscale
- node clusters (required structure):
    - (none resolved)
- optional roles: GetVideoComponents, MarkdownNote, SaveVideo, ByteDanceVideoEnhanceNode, ClarityCrystalUpscaleNode, ClarityCrystalVideoUpscaleNode, CreateVideo, HitPawGeneralImageEnhance, HitPawVideoEnhance, ImageCompare, ImageStitch, LoadImage
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Upscale / Topaz  (`api_partner_nodes_upscale__topaz`)  -  6 workflow(s)  -  source: mixed
- execution: api (API nodes: TopazImageEnhance, TopazImageEnhanceV2, TopazVideoEnhance)
- when to use: Use to upscale / enhance an image using Topaz.
- example request: "build an image workflow using Topaz"
- description: API image enhancement/upscaling via Topaz Reimagine. 1 image -> 1 enhanced image output. Applies face enhancement and detail restoration for professional results. | API video upscaling via Topaz AI. 1 video -> 1 enhanced video output. Supports resolution upscaling (Starlight/Astra Fast model) and frame interpolation (apo-8 model). | Enhance AI-generated images with Bloom 2, a creative upscaling model that adds new detail, texture, and visual elements while offering a Creativity control from 1-9 for subtle refinement or bold reinterpretation. It accepts 1 input image and produces 1 enhanced output, with optional Face Recovery, Color Preservation, and grain settings for fine-tuned results. Ideal for improving AI artwork and illustrations, adding realism to CG renders, and refining facial detail in generated portraits. | Upload a landscape image to upscale and enhance details. Generate a high-resolution output optimized for natural scenery using Topaz Reimagine. | Upload an illustration to upscale using Topaz. Generate a smooth, stylized image with accurate colors, though some artifacts may appear. | Upscale and enhance a single image using Topaz's Wonder 3.5 generative model, which adds fine detail, reduces noise, and sharpens text and structured graphics while offering Low, Medium, and High enhancement strengths. The workflow takes 1 input image and produces 1 enhanced output alongside a side-by-side comparison view. Ideal for restoring old or compressed photos, improving product shots and marketing visuals, and preparing low-resolution images for print or large-format display.
- member workflows:
    - api_topaz_image_enhance
    - api_topaz_image_enhance_bloom2
    - api_topaz_image_enhance_wonder3_5
    - api_topaz_video_enhance
    - utility-topaz_landscape_upscaler
    - utility_topaz_illustration_upscale
- node clusters (required structure):
    - (none resolved)
- optional roles: MarkdownNote, ImageCompare, ImageScaleBy, LoadImage, LoadVideo, SaveImage, SaveImageAdvanced, SaveVideo, TopazImageEnhance, TopazImageEnhanceV2, TopazVideoEnhance
- unresolved nodes: MarkdownNote, PrimitiveNode

## API / Partner Nodes - Upscale / Magnific  (`api_partner_nodes_upscale__magnific`)  -  3 workflow(s)  -  source: mixed
- execution: api (API nodes: MagnificImageSkinEnhancerNode, MagnificImageUpscalerCreativeNode, MagnificImageUpscalerPreciseV2Node)
- when to use: Use to upscale / enhance an image using Magnific.
- example request: "build an image workflow using Magnific"
- description: API creative image upscaling via Magnific. 1 image -> 1 upscaled image output. Supports up to 16x enlargement with creative detail enhancement. | API precise image upscaling via Magnific. 1 image -> 1 high-resolution image output. Upscales with strict detail preservation and enhanced sharpness. | Enhance skin and overall image quality by uploading your image and selecting a preset to adjust lighting, color grading, and realism.
- member workflows:
    - api_magnific_image_upscale_creative
    - api_magnific_image_upscale_precise
    - api_magnific_skin_enhancer
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
- optional roles: ImageCompare, MagnificImageSkinEnhancerNode, MagnificImageUpscalerCreativeNode, MagnificImageUpscalerPreciseV2Node

## API / Partner Nodes - Upscale / Nano-Banana  (`api_partner_nodes_upscale__nano_banana`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: GeminiImage2Node)
- when to use: Use to upscale / enhance an image using Nano-Banana.
- example request: "build an image workflow using Nano-Banana"
- description: Upload a product image to enhance its detail and sharpness. Generate an upscaled, high-quality output with improved typography and realistic coherence using the Nano Banana Pro model. | Upload a stylized art or illustration. Generate a creatively upscaled 4K image with enhanced detail using the Nano Banana Pro model, ideal for marketing or gaming assets.
- member workflows:
    - utility_nanobanana_pro_illustration_upscale
    - utility_nanobanana_pro_product_upscale
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: GeminiImage2Node

## API / Partner Nodes - Upscale / Flux 3  (`api_partner_nodes_upscale__flux_3`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: FluxVideoUpscaleNode)
- when to use: Use to upscale / enhance a video using Flux 3.
- example request: "build a video workflow using Flux 3"
- description: Upscale video clips to 1080p, 2K, or 4K using FLUX 3 super-resolution, with a precise mode that preserves source fidelity and a creative mode that enhances fine detail. It accepts 1 video input, up to 20 seconds long, and produces 1 upscaled mp4 output with an optional prompt to guide detail enhancement. Ideal for restoring low-resolution footage, preparing clips for large-screen delivery, and adding crispness to AI-generated video.
- member workflows:
    - api_bfl_flux_video_upscale
- node clusters (required structure):
    - inputs: LoadVideo
    - output: SaveVideo
    - other operations: FluxVideoUpscaleNode, MarkdownNote
- unresolved nodes: MarkdownNote


# Preprocessors / Estimation  (`preprocessors_estimation`)  -  21 workflow(s), 8 model(s)

## Preprocessors / Estimation / MoGe  (`preprocessors_estimation__moge`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using MoGe.
- example request: "build an image workflow using MoGe"
- description: Estimate fine-detailed geometry from a single image with MoGe 3, the sparse-volumetric refinement version of MoGe. It lifts a coarse point map onto a sparse voxel shell and refines depth with lightweight sparse 3D convolutions, keeping thin structures and depth discontinuities that image-space decoders blur. Ships with the ViT-L checkpoint; the ViT-G file is selectable in the same Load MoGe Model dropdown for higher fine-detail accuracy. Takes 1 image input and outputs 1 depth map and 1 normal map. | Estimates monocular depth from an input image using MoGe, outputting both raw and colorized depth maps plus a mask. | Estimates monocular depth from an input video using MoGe, outputting both raw and colorized depth maps plus a mask. | Upload a single RGB image and adjust inference resolution and batch size. Generate a colored depth preview and raw depth map,
- member workflows:
    - image_depth_estimation_moge
    - utility_moge3_geometry_estimation
    - utility_moge_depth_estimation
    - video_depth_estimation_moge
- node clusters (required structure):
    - other operations: LoadMoGeModel, MoGeInference, MoGeRender
- optional roles: MarkdownNote, PreviewImage, GetVideoComponents, ImageToMask, LoadImage, MaskPreview, ResizeImageMaskNode, ResizeImagesByLongerEdge
- unresolved nodes: MarkdownNote

## Preprocessors / Estimation / SDPose  (`preprocessors_estimation__sdpose`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a pose map using SDPose.
- example request: "build an image workflow using SDPose"
- description: Detects multiple people in an image and outputs per-person pose keypoints, skeleton renders, and bounding boxes using SDPose. | Extracts human pose keypoints and stick-figure visuals from an image using SDPose-OOD, with optional bounding-box input per subject. | Extracts multi-person pose keypoints and skeleton frame sequences from video using SDPose with built-in person detection. | Upload an image to extract pose keypoints and generate a corresponding pose map using the SDPose-OOD model.
- member workflows:
    - image_to_pose_map_sdpose_multi_person
    - image_to_pose_map_sdpose_ood
    - utility_sdpose_ood_image_to_pose
    - video_to_pose_map_sdpose_multi_person
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - other operations: ResizeImageMaskNode, SDPoseDrawKeypoints, SDPoseKeypointExtractor
- optional roles: GetVideoComponents, LoadImage, MarkdownNote, RTDETR_detect, SaveImage, UNETLoader
- unresolved nodes: MarkdownNote

## Preprocessors / Estimation / Depth Anything  (`preprocessors_estimation__depth_anything`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using Depth Anything.
- example request: "build an image workflow using Depth Anything"
- description: This subgraph processes a video input through Depth Anything 3 to produce temporally consistent depth maps for each frame, outputting a depth video. It is ideal for video content requiring spatial geometry estimation, such as 3D reconstruction, SLAM, or novel view synthesis from moving cameras. The model uses a plain transformer backbone trained with a depth-ray representation, supporting any number of views without requiring known camera poses. | This subgraph takes an input image and produces a depth map using the Depth Anything 3 model, which recovers spatially consistent geometry from any number of views. It is ideal for single or multi-view images, videos, and 3D scenes where accurate depth estimation is needed for tasks like SLAM, novel view synthesis, or spatial perception. The model uses a plain transformer backbone and supports both monocular and multi-view inputs without. | Upload 1 image. Generate a depth map using Depth Anything 3 and view a side-by-side comparison of the original and depth output. Ideal for 3D scene reconstruction, AR/VR asset preparation, and visual effects depth passes.
- member workflows:
    - image_depth_estimation_depth_anything_3
    - utility_depth_anything3_image_depth_estimation
    - video_depth_estimation_depth_anything_3
- node clusters (required structure):
    - other operations: DA3Inference, DA3Render, LoadDA3Model
- optional roles: MarkdownNote, GetVideoComponents, ImageCompare, LoadImage, PreviewImage, Video Slice
- unresolved nodes: MarkdownNote

## Preprocessors / Estimation / Qwen Image  (`preprocessors_estimation__qwen_image`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Estimate albedo from an input image using Marigold V2. The workflow separates surface color from lighting and shading, producing an albedo map useful for relighting, material analysis, and 3D asset creation. | Estimate scene depth from an input image using Marigold V2, generating a depth map with brighter areas representing closer objects and darker areas representing farther objects. | Estimate surface normals from an input image using Marigold V2. The workflow generates a normal map representing the orientation of surfaces in the scene, useful for 3D reconstruction, relighting, and geometry-aware image processing.
- member workflows:
    - image_marigold_v2_albedo_estimation
    - image_marigold_v2_depth_estimation
    - image_marigold_v2_surface_normal_estimation
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicGuider, ConditioningLoader, ManualSigmas
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - output: SaveImageAdvanced
    - other operations: DisableNoise, ImageCompare, MarigoldV2PostProcess, MarkdownNote, ModelSamplingAuraFlow
- unresolved nodes: MarkdownNote

## Preprocessors / Estimation / Lotus  (`preprocessors_estimation__lotus`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using Lotus.
- example request: "build an image workflow using Lotus"
- description: Estimates a monocular depth map from an input image using the Lotus depth estimation model. | Image to Depth Map (Lotus) blueprint
- member workflows:
    - image_depth_estimation_lotus_depth
    - image_to_depth_map_lotus
- node clusters (required structure):
    - model loading: UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, LotusConditioning
    - latent / canvas: VAEEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode
    - other operations: DisableNoise, ImageInvert, SetFirstSigma

## Preprocessors / Estimation / MediaPipe  (`preprocessors_estimation__mediapipe`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to detect facial landmarks using MediaPipe.
- example request: "build an image workflow using MediaPipe"
- description: Detects facial landmarks from a video using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask. | Detects facial landmarks from an image using MediaPipe, outputting landmark data, face bounding boxes, and an optional face-region mask.
- member workflows:
    - image_face_detection_mediapipe
    - video_face_detection_mediapipe
- node clusters (required structure):
    - other operations: LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceMask
- optional roles: GetVideoComponents, Video Slice

## Preprocessors / Estimation / SAM3  (`preprocessors_estimation__sam3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to segment an image using SAM3.
- example request: "build an image workflow using SAM3"
- description: Segments images into masks using Meta SAM3 from text prompts, points, or boxes. | Segments video into temporally consistent masks using Meta SAM3 from text or interactive prompts.
- member workflows:
    - image_segmentation_sam3
    - video_segmentation_sam3
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode
    - other operations: SAM3_Detect
- optional roles: GetVideoComponents, Note
- unresolved nodes: Note

## Preprocessors / Estimation / BiRefNet  (`preprocessors_estimation__birefnet`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to remove the background from an image using BiRefNet.
- example request: "build an image workflow using BiRefNet"
- description: Upload an image with any background. Generate a version with the background removed and a precision segmentation mask.
- member workflows:
    - utility_birefnet_remove_background
- node clusters (required structure):
    - inputs: LoadImage
    - output: PreviewImage
    - other operations: InvertMask, JoinImageWithAlpha, LoadBackgroundRemovalModel, MarkdownNote, MaskPreview, RemoveBackground
- unresolved nodes: MarkdownNote


# Audio  (`audio`)  -  18 workflow(s), 5 model(s)

## Audio / ACE-Step  (`audio__ace_step`)  -  10 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using ACE-Step.
- example request: "build an audio workflow using ACE-Step"
- description: Edit existing songs to change style and lyrics using ACE-Step v1 M2M. | Generate high-quality audio from text prompts using the 4B parameter ACE-Step 1.5 XL model. | Generate high-quality music from text prompts using a 4B parameter SFT model. Control output with CFG for precise prompt adherence. | Generate high-quality music from text prompts using the distilled 4B ACE-Step model. Produces commercial-ready audio in just 8 inference steps without CFG. | Generate instrumental music from text prompts using ACE-Step v1. | Generate songs with vocals from text prompts using ACE-Step v1, supporting multilingual and style customization. | Generates audio/music from text prompts using ACE-Step 1.5, a diffusion-based audio generation model. | Input a text prompt describing the music style and optional lyrics. Generate a full, high-quality audio song in under 10 seconds on consumer hardware. | Input a text prompt to generate music. This 4B model version offers stronger audio understanding and composition capabilities compared to smaller variants. | Input style tags and lyrics to generate a full song. The workflow uses the ACE-Step 1.5 model to produce commercial-grade music in under 10 seconds on consumer hardware.
- member workflows:
    - audio_ace_step1_5_xl_base
    - audio_ace_step1_5_xl_sft
    - audio_ace_step1_5_xl_turbo
    - audio_ace_step_1_5_checkpoint
    - audio_ace_step_1_5_split
    - audio_ace_step_1_5_split_4b
    - audio_ace_step_1_m2m_editing
    - audio_ace_step_1_t2a_instrumentals
    - audio_ace_step_1_t2a_song
    - text_to_audio_ace_step_1_5
- node clusters (required structure):
    - conditioning: ConditioningZeroOut
    - sampling: KSampler
    - decoding: VAEDecodeAudio
- optional roles: CheckpointLoaderSimple, DualCLIPLoader, EmptyAceStep1.5LatentAudio, EmptyAceStepLatentAudio, LatentApplyOperationCFG, LatentOperationTonemapReinhard, LoadAudio, MarkdownNote, ModelSamplingAuraFlow, ModelSamplingSD3, Note, SaveAudioAdvanced
- unresolved nodes: MarkdownNote, Note, PrimitiveNode

## Audio / Qwen Image  (`audio__qwen_image`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using Qwen Image, Stable Audio.
- example request: "build an audio workflow using Qwen Image"
- description: Generates music, instrument loops, sound effects, and one-shots from text using Stable Audio 3 Medium, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot). | Generates music, instrument loops, sound effects, and one-shots from text using the Stable Audio 3 Medium base checkpoint, with optional Qwen 3.5 category-based prompt expansion (Music, Instrument, SFX, One-shot). | Input a short text description of a sound, music, or effect. The workflow expands your prompt with Qwen and generates a stereo audio clip from Stable Audio 3.
- member workflows:
    - audio_generation_stable_audio_3_medium
    - audio_generation_stable_audio_3_medium_base
    - audio_stable_audio_3_medium_base
- node clusters (required structure):
    - model loading: CLIPLoader (x2), CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentAudio
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - other operations: ComfyMathExpression, ComfySwitchNode, CustomCombo, JsonExtractString, PreviewAny (x2), PrimitiveBoolean, PrimitiveFloat, PrimitiveStringMultiline, StringReplace (x3), TextGenerate
- paired/multiple required: CLIPLoader x2, CLIPTextEncode x2
- optional roles: MarkdownNote, SaveAudioAdvanced
- unresolved nodes: MarkdownNote

## Audio / Generic  (`audio__generic`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt.
- example request: "build an audio workflow"
- description: Create a new version of a song from reference audio with YuE2. Preserve the original melody while changing the musical style, vocals, lyrics, and arrangement. | Generate complete songs from a music style and lyrics with YuE2. Use symbolic music planning to create structured melodies and harmonies, then turn them into full songs with vocals and instrumental accompaniment.
- member workflows:
    - audio_yue2_music_cover
    - audio_yue2_text2music
- node clusters (required structure):
    - model loading: CheckpointLoaderSimple
    - conditioning: ConditioningZeroOut
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - output: SaveAudioAdvanced
    - other operations: EmptyYuE2LatentAudio, MarkdownNote, PreviewAny (x2), SeedNode, YuE2GenerateMusic
- optional roles: AudioEncoderLoader, LoadAudio, SheetSage2AudioToABC, YuE2GenerateABC
- unresolved nodes: MarkdownNote

## Audio / Stable Audio  (`audio__stable_audio`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using Stable Audio.
- example request: "build an audio workflow using Stable Audio"
- description: Generate audio from text prompts using Stable Audio. | Input a short text idea, optional duration, seed, and category. Generate stereo audio (music, SFX, or instruments) using Stable Audio 3 with optional AI-driven text expansion.
- member workflows:
    - audio_stable_audio_3_medium
    - audio_stable_audio_example
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentAudio
    - sampling: KSampler
    - decoding: VAEDecodeAudio
    - output: SaveAudioAdvanced
    - other operations: MarkdownNote
- paired/multiple required: CLIPTextEncode x2
- optional roles: CustomCombo, TextGenerate
- unresolved nodes: MarkdownNote

## Audio / MiniMax  (`audio__minimax`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate audio from a text prompt using MiniMax.
- example request: "build an audio workflow using MiniMax"
- description: Create full-length songs from text prompts with MiniMax Music 3, generating complete 5-minute tracks with stable structure and high-fidelity audio. Input a structured caption describing lyrics, style, mood, and arrangement, and receive a finished song as output. Ideal for music producers prototyping demos, and content creators needing original background tracks.
- member workflows:
    - audio_minimax_music_3
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: ConditioningZeroOut, MiniMaxMusic3TextEncode
    - sampling: KSampler
    - decoding: VAEDecodeAudio, VAEDecodeAudioTiled
    - output: SaveAudioAdvanced
    - other operations: ComfySwitchNode, EmptyMiniMaxMusic3LatentAudio, MarkdownNote (x2), SeedNode
- paired/multiple required: MarkdownNote x2
- unresolved nodes: MarkdownNote


# Image Edit with ControlNet  (`image_edit_with_controlnet`)  -  18 workflow(s), 7 model(s)

## Image Edit with ControlNet / Qwen Image  (`image_edit_with_controlnet__qwen_image`)  -  5 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Control image generation using Qwen-Image ControlNet models. Supports canny, depth, and inpainting controls through model patching. | Describe the elements you want in the image and decompose it precisely from the image. | Generate images with Qwen-Image InstantX ControlNet, supporting canny, soft edge, depth, pose | Generate images with precise structural control using Qwen-Image's unified ControlNet LoRA. Supports multiple control types including canny, depth, lineart, softedge, normal, and openpose for diverse creative applications. | Upload an image and select a control type from Canny, HED, Depth, Pose, MLSD, Scribble, or Grayscale. Generate a new image guided by the chosen structural condition.
- member workflows:
    - image_qwen_Image_2512_controlnet
    - image_qwen_image_controlnet_patch
    - image_qwen_image_instantx_controlnet
    - image_qwen_image_layered_control
    - image_qwen_image_union_control_lora
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: MarkdownNote (x2), ModelSamplingAuraFlow
- paired/multiple required: MarkdownNote x2, CLIPTextEncode x2
- optional roles: LoraLoaderModelOnly, Note, ReferenceLatent, VAEEncode, BasicGuider, BasicScheduler, Canny, ControlNetApplyAdvanced, ControlNetLoader, DisableNoise, EmptyQwenImageLayeredLatentImage, EmptySD3LatentImage
- unresolved nodes: MarkdownNote, Note, Reroute

## Image Edit with ControlNet / Z-Image  (`image_edit_with_controlnet__z_image`)  -  5 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Generates an image from a Canny edge map using Z-Image-Turbo, with text conditioning. | Generates an image from a depth map using Z-Image-Turbo with text conditioning. | Generates an image from pose keypoints using Z-Image-Turbo with text conditioning. | Generates images from a text prompt and ControlNet conditioning (e.g. depth, canny) using Z-Image-Turbo. | [Local] image editing via Z-Image-Turbo. 1 image input -> 1 image output. Uses ControlNet for precise and controlled image editing.
- member workflows:
    - canny_to_image_z_image_turbo
    - controlnet_z_image_turbo
    - depth_to_image_z_image_turbo
    - image_z_image_turbo_fun_union_controlnet
    - pose_to_image_z_image_turbo
- node clusters (required structure):
    - model loading: CLIPLoader, ModelPatchLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, QwenImageDiffsynthControlnet
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: GetImageSize, ModelSamplingAuraFlow
- optional roles: BasicGuider, BasicScheduler, Canny, DisableNoise, ImageInvert, ImageScaleToTotalPixels, KSamplerSelect, LoadImage, LotusConditioning, PreviewImage, SamplerCustomAdvanced, SaveImage

## Image Edit with ControlNet / Anima  (`image_edit_with_controlnet__anima`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Anima.
- example request: "build an image workflow using Anima"
- description: Apply lightweight, LoRA-like conditional control to Anima-Base v1.0 anime-style images using ControlNet-LLLite, accepting one conditioning input and one optional mask for inpainting to produce a single generated output. Ideal for anime inpainting with dynamic masking, style transfer from lineart or grayscale sketches, and controlled character generation in multi-scene compositions. | Generate anime-style illustrations from a depth map using the Anima LLLite model, which applies a lightweight ControlNet correction for precise spatial control. Input 1 image to produce 2 image outputs plus a comparison view. Ideal for 3D scene reconstruction, VFX depth passes, and AR asset preparation.
- member workflows:
    - image_anima_lllite_any_control_to_image
    - image_anima_lllite_depth_control_to_image
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, LoraLoaderModelOnly, ModelPatchLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage, SaveImage
    - other operations: AnimaLLLiteApply, ComfySwitchNode (x3), GetImageSize, ImageCompare, MarkdownNote, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x2), ResizeImageMaskNode
- paired/multiple required: CLIPTextEncode x2
- optional roles: Canny, DA3Inference, DA3Render, ImageInvert, LoadDA3Model
- unresolved nodes: MarkdownNote

## Image Edit with ControlNet / Flux  (`image_edit_with_controlnet__flux`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using Flux.
- example request: "build an image workflow using Flux"
- description: Generate images guided by edge detection using Flux.1 Canny. | [Local] image-to-image via Flux Redux with ControlNet Union. Image/video + prompt -> image. Style and content transfer guided by Redux style model and ControlNet hints.
- member workflows:
    - bepic_templates__bEpic_FluxRedux
    - flux_canny_model_example
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, FluxGuidance
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
- optional roles: SetNode, GetNode, bEpicReformat, ControlNetApplyAdvanced, ControlNetLoader, SetUnionControlNetType, VHS_LoadImagePath, VHS_SelectImages, AIO_Preprocessor, CLIPVisionEncode, CLIPVisionLoader, Canny
- unresolved nodes: GetNode, MarkdownNote, SetNode, easy cleanGpuUsed, easy imageSplitGrid

## Image Edit with ControlNet / Generic  (`image_edit_with_controlnet__generic`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image.
- example request: "build an image workflow"
- description: Generate images guided by blurred reference images using SD 3.5. | Generate images guided by edge detection using SD 3.5 Canny ControlNet.
- member workflows:
    - sd3.5_large_blur
    - sd3.5_large_canny_controlnet_example
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple
    - conditioning: CLIPTextEncode, ConditioningZeroOut, ControlNetApplyAdvanced, ControlNetLoader
    - latent / canvas: EmptySD3LatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: MarkdownNote
- optional roles: Canny, ImageScale, PreviewImage
- unresolved nodes: MarkdownNote

## Image Edit with ControlNet / Lotus  (`image_edit_with_controlnet__lotus`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Lotus.
- example request: "build an image workflow using Lotus"
- description: Generate images guided by depth information using SD 3.5.
- member workflows:
    - sd3.5_large_depth
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CheckpointLoaderSimple, UNETLoader, VAELoader
    - conditioning: BasicGuider, BasicScheduler, CLIPTextEncode, ConditioningZeroOut, ControlNetApplyAdvanced, ControlNetLoader, LotusConditioning
    - latent / canvas: EmptySD3LatentImage, VAEEncode (x2)
    - sampling: KSampler, KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode (x2)
    - output: PreviewImage, SaveImage
    - other operations: DisableNoise, ImageInvert, ImageScaleToTotalPixels, MarkdownNote (x3), SetFirstSigma
- paired/multiple required: MarkdownNote x3, VAEDecode x2, VAEEncode x2
- unresolved nodes: MarkdownNote

## Image Edit with ControlNet / MoGe  (`image_edit_with_controlnet__moge`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image guided by a control map (canny/depth/pose) using MoGe.
- example request: "build an image workflow using MoGe"
- description: Re-pose and re-style a reference photo with MOSS-Image: the graph estimates the reference's geometry with MoGe 3, then generates a new image that keeps the pose, framing and silhouette while the prompt decides the character, wardrobe, setting and light. A color/depth control adapter LoRA makes the model follow the estimated map, and the color map socket is there for region-painted layouts.
- member workflows:
    - image_moss_image_color_depth_control
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: BasicScheduler, CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentImage
    - sampling: SamplerCustom, SamplerER_SDE
    - decoding: VAEDecode
    - output: PreviewImage (x2), SaveImage
    - other operations: GetImageSize, ImageCompare, ImageScaleToTotalPixels, LoadMoGeModel, MOSSImageControl, MarkdownNote (x2), MoGeInference, MoGeRender, ModelSamplingAuraFlow, Painter, SkipLayerGuidanceDiTSimple, TemporalScoreRescaling
- paired/multiple required: CLIPTextEncode x2, MarkdownNote x2, PreviewImage x2
- unresolved nodes: MarkdownNote


# Text to Video  (`text_to_video`)  -  18 workflow(s), 8 model(s)

## Text to Video / LTX-2  (`text_to_video__ltx_2`)  -  5 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate a video from a text prompt, optionally using an image for reference. Receive a high-quality video with improved motion, audio, and detail, optimized for portrait or landscape formats. | Generate cinematic video from a text prompt using LTX-2.5, a production-ready diffusion model that outputs a single video file with optional spatial upscaling for enhanced resolution. Ideal for film previsualization, commercial storyboards, and high-volume content creation pipelines. | Generate high-quality videos from text prompts with synchronized audio-video generation using LTX-2 distilled model. Optimized for faster generation while maintaining quality. Features expressive lip sync, dynamic motion generation, and improved speed. | Generate high-quality videos from text prompts with synchronized audio-video generation using LTX-2. Features expressive lip sync, dynamic motion generation, and efficient performance. | Generates video from text prompts using LTX-2.3, Lightricks' video diffusion model.
- member workflows:
    - text_to_video_ltx_2_3
    - video_ltx2_3_t2v
    - video_ltx2_5_t2v
    - video_ltx2_t2v
    - video_ltx2_t2v_distilled
- node clusters (required structure):
    - model loading: LatentUpscaleModelLoader
    - conditioning: CLIPTextEncode, LTXVConditioning, ManualSigmas
    - latent / canvas: LTXVEmptyLatentAudio
    - sampling: KSamplerSelect (x2), LTXVLatentUpsampler, SamplerCustomAdvanced (x2)
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - other operations: CreateVideo, EmptyLTXVLatentVideo, LTXVConcatAVLatent (x2), LTXVSeparateAVLatent (x2), PrimitiveInt (x2), RandomNoise (x2)
- paired/multiple required: KSamplerSelect x2, LTXVConcatAVLatent x2, LTXVSeparateAVLatent x2, RandomNoise x2, SamplerCustomAdvanced x2
- optional roles: MarkdownNote, LoraLoaderModelOnly, CFGGuider, CLIPLoader, LTXVDualCFGGuider, LTXVImgToVideoInplace, ResizeImageMaskNode, VAELoader, CheckpointLoaderSimple, EmptyImage, ImageScaleBy, LTXAVTextEncoderLoader
- unresolved nodes: MarkdownNote, Reroute

## Text to Video / Generic  (`text_to_video__generic`)  -  4 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video from a text prompt.
- example request: "build a video workflow"
- description: A lightweight 2B model that generates videos from English and Russian prompts with high visual quality. | Generate synchronized video and audio directly from text using a distilled eight-step model, producing one video output with no file inputs required. Ideal for rapid text-to-video prototyping, audio-visual content creation, and creative storytelling where fast generation matters more than maximum detail. | [Local] video decomposition via UniVidX. Frames -> extract alpha mattes and intrinsic channels. Use when you need to separate foreground motion from static backgrounds for compositing workflows. | [Local] video export via VHS_VideoCombine and UniVidX. {} -> 4 videos. Pick this over the base t2RAIN graph for ready MP4s.
- member workflows:
    - UniVidX_ComfyUI__I_video_output
    - UniVidX_ComfyUI__I_video_output_api
    - video_fastvideo_fasth3_t2v
    - video_kandinsky5_t2v
- node clusters (required structure):
    - (none resolved)
- optional roles: VHS_VideoCombine, CLIPTextEncode, MarkdownNote, VAELoader, BasicGuider, BasicScheduler, BlockSparseAttention, CLIPLoader, CreateVideo, DualCLIPLoader, KSampler, KSamplerSelect
- unresolved nodes: MarkdownNote, Note

## Text to Video / Hunyuan Video  (`text_to_video__hunyuan_video`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using Hunyuan Video.
- example request: "build a video workflow using Hunyuan Video"
- description: Generate high-quality 720p videos from text prompts with cinematic camera control, emotional expressions, and physics simulation. Supports multiple styles including realistic, anime, and 3D with text rendering. | Generate videos from text prompts using Hunyuan model.
- member workflows:
    - hunyuan_video_text_to_video
    - video_hunyuan_video_1.5_720p_t2v
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: BasicScheduler, CLIPTextEncode
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode, VAEDecodeTiled
    - output: SaveVideo
    - other operations: CreateVideo, MarkdownNote, ModelSamplingSD3, Note (x2), RandomNoise
- paired/multiple required: Note x2
- optional roles: CFGGuider, EasyCache, BasicGuider, DisableNoise, EmptyHunyuanLatentVideo, EmptyHunyuanVideo15Latent, FluxGuidance, HunyuanVideo15LatentUpscaleWithModel, HunyuanVideo15SuperResolution, LatentUpscaleModelLoader, SplitSigmas
- unresolved nodes: MarkdownNote, Note

## Text to Video / WAN  (`text_to_video__wan`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using WAN.
- example request: "build a video workflow using WAN"
- description: Generate text-to-video with alpha channel support for transparent backgrounds and semi-transparent objects. | Generate videos from text prompts using Wan 2.1.
- member workflows:
    - text_to_video_wan
    - video_wan2.1_alpha_t2v_14B
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: EmptyHunyuanLatentVideo, ModelSamplingSD3
- paired/multiple required: CLIPTextEncode x2
- optional roles: LoraLoaderModelOnly, MarkdownNote, CreateVideo, ImageToMask, InvertMask, JoinImageWithAlpha, SaveAnimatedWEBP, SaveVideo
- unresolved nodes: MarkdownNote

## Text to Video / WAN 2.2  (`text_to_video__wan_2_2`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Generate high-quality videos from text prompts with cinematic aesthetic control and dynamic motion generation using Wan 2.2. | Generates video from text prompts using Wan2.2, Alibaba's diffusion video model.
- member workflows:
    - text_to_video_wan_2_2
    - video_wan2_2_14B_t2v
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly (x2), UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, EmptyHunyuanLatentVideo, MarkdownNote, ModelSamplingSD3 (x2)
- paired/multiple required: CLIPTextEncode x2, KSamplerAdvanced x2, LoraLoaderModelOnly x2, ModelSamplingSD3 x2, UNETLoader x2
- optional roles: Note, SaveVideo
- unresolved nodes: MarkdownNote, Note

## Text to Video / LTX-Video  (`text_to_video__ltx_video`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using LTX-Video.
- example request: "build a video workflow using LTX-Video"
- description: Generate videos from text prompts.
- member workflows:
    - ltxv_text_to_video
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple
    - conditioning: CLIPTextEncode (x2), LTXVConditioning
    - sampling: KSamplerSelect, SamplerCustom
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, EmptyLTXVLatentVideo, LTXVScheduler, MarkdownNote, Note
- paired/multiple required: CLIPTextEncode x2
- unresolved nodes: MarkdownNote, Note

## Text to Video / MiniMax  (`text_to_video__minimax`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using MiniMax.
- example request: "build a video workflow using MiniMax"
- description: Generate video with native stereo audio directly from a text prompt using MiniMax H3, an omni-modal model that synthesizes voice, sound effects, and music in a single pass. Output is up to 2K resolution, 24fps, and roughly 15 seconds long, with no file inputs required - just a prompt to one video output. Ideal for rapid concept visualization, social media content creation, and audio-synced storytelling.
- member workflows:
    - video_minimax_h3_t2v
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader (x2)
    - conditioning: BasicGuider, BasicScheduler
    - sampling: KSamplerSelect, SamplerCustomAdvanced
    - decoding: VAEDecode, VAEDecodeAudio
    - output: SaveVideo
    - other operations: ComfyMathExpression, ComfySwitchNode (x2), CreateVideo, MarkdownNote (x3), MiniMaxH3ImageToVideo, PrimitiveBoolean, PrimitiveFloat, PrimitiveInt (x2), RandomNoise, ResolutionSelector
- paired/multiple required: MarkdownNote x3, VAELoader x2
- unresolved nodes: MarkdownNote

## Text to Video / WAN VACE  (`text_to_video__wan_vace`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video from a text prompt using WAN VACE.
- example request: "build a video workflow using WAN VACE"
- description: Transform text descriptions into high-quality videos. Supports both 480p and 720p with VACE-14B model.
- member workflows:
    - video_wan_vace_14B_t2v
- node clusters (required structure):
    - model loading: CLIPLoader (x2), LoraLoader (x2), UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, MarkdownNote (x5), ModelSamplingSD3, SaveAnimatedWEBP, TrimVideoLatent, WanVaceToVideo
- paired/multiple required: MarkdownNote x5, CLIPLoader x2, CLIPTextEncode x2, LoraLoader x2, UNETLoader x2
- unresolved nodes: MarkdownNote


# 3D  (`3d`)  -  15 workflow(s), 4 model(s)

## 3D / Hunyuan3D  (`3d__hunyuan3d`)  -  5 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a 3D model using Hunyuan3D.
- example request: "build a 3d workflow using Hunyuan3D"
- description: Generate 3D models from multiple views using Hunyuan3D 2.0 MV Turbo. | Generate 3D models from multiple views using Hunyuan3D 2.0 MV. | Generate a textured-ready 3D mesh from a single image using Hunyuan3D 2.0's shape diffusion pipeline, which aligns geometry to the given image and outputs an untextured white mesh. Ideal for 3D asset prototyping, game and AR content prep, and as a base for later texture painting or handcrafted detailing. | Generates 3D mesh models from a single input image using Hunyuan3D 2.0/2.1. | Reconstruct a 3D model from a single photo using Hunyuan3D 2.1, converting one input image into a raw white-mesh geometry without textures or materials. Ideal for rapid 3D concepting, blockout prototyping, and asset preparation before texturing or retopology.
- member workflows:
    - 3d_hunyuan3d-v2.1
    - 3d_hunyuan3d_image_to_model
    - 3d_hunyuan3d_multiview_to_model
    - 3d_hunyuan3d_multiview_to_model_turbo
    - image_to_model_hunyuan3d_2_1
- node clusters (required structure):
    - model loading: ImageOnlyCheckpointLoader
    - latent / canvas: EmptyLatentHunyuan3Dv2
    - sampling: KSampler
    - decoding: VAEDecodeHunyuan3D
    - other operations: CLIPVisionEncode, ModelSamplingAuraFlow, VoxelToMesh
- optional roles: LoadImage, MarkdownNote, FluxGuidance, Hunyuan3Dv2Conditioning, Hunyuan3Dv2ConditioningMultiView, SaveGLB
- unresolved nodes: MarkdownNote

## 3D / Generic  (`3d__generic`)  -  4 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate an image.
- example request: "build an image workflow"
- description: [Local] gaussian-splatting via SHARP model. image and exif data -> ply files. Uses camera focal length metadata to reconstruct scene geometry from a single photo. | [Local] generate 3D Gaussians via Apple SHARP. single image -> four preview frames. Pick it for rapid monocular 3D scene creation. | [Local] image-to-3D via SHARP. Image -> PLY Gaussian splats. Pick for instant monocular 3D reconstruction from a single photo. | [Local] image-to-3d-gaussians via SHARP. image -> ply. Compare standard 3D gaussian generation against results aligned using external metric depth as a geometric scaffold.
- member workflows:
    - ComfyUI-Sharp__pano
    - ComfyUI-Sharp__sharp_basic
    - ComfyUI-Sharp__sharp_exif
    - ComfyUI-Sharp__vanilla_vs_image_depth_gaussians
- node clusters (required structure):
    - other operations: LoadSharpModel
- optional roles: PreviewImage, PreviewGaussianSharp, DownloadAndLoadMoGe2Model, ImageFromBatch, LoadImage, LoadImageWithExif, MergeGaussians, MoGe2Inference, PanoramaDepthMerge, PanoramaSplit, PanoramaSplitAdaptive, PanoramaWrap
- unresolved nodes: DownloadAndLoadMoGe2Model, MoGe2Inference, PanoramaDepthMerge, PanoramaSplit, PanoramaSplitAdaptive, PanoramaWrap, PreviewGaussianSharp, PreviewGaussians, TransformGaussian

## 3D / MoGe  (`3d__moge`)  -  4 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using MoGe.
- example request: "build a 3d workflow using MoGe"
- description: Estimates 3D scene geometry from an input image using MoGe, outputting a mesh plus OpenGL and DirectX normal maps. | Upload a single image. Generate a high-fidelity 3D model with PBR textures using either Pixal3D or TRELLIS.2, toggled by a single switch. Both pipelines share the same DINOv3 encoder and shape/texture VAEs; Pixal3D adds MoGe camera estimation for pixel-aligned geometry. Includes background removal and a complete mesh post-processing pipeline: remesh, decimate, UV unwrap, and texture baking. | Upload an equirectangular 360° panorama image and generate a textured GLB mesh with vertex colors. | Upload an image to estimate its perspective geometry. Generate a 3D depth map and surface normals from the input.
- member workflows:
    - 3d_moge_panorama_to_mesh
    - 3d_moge_perspective_to_mesh
    - 3d_pixal3d_trellis2_image_to_model
    - geometry_estimation_moge
- node clusters (required structure):
    - other operations: ComfySwitchNode (x2), LoadMoGeModel
- optional roles: PreviewImage, KSampler, MeshToFile3D, CFGOverride, MarkdownNote, MeshSmoothNormals, MoGeRender, Preview3DAdvanced, RescaleCFG, UNETLoader, VAELoader, ApplyTextureToMesh
- unresolved nodes: MarkdownNote, Note

## 3D / TripoSplat  (`3d__triposplat`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a 3D model using TripoSplat.
- example request: "build a 3d workflow using TripoSplat"
- description: This subgraph takes a single 2D image as input and generates a variable number of 3D Gaussians (up to 262,144) as output, enabling high-quality 3D reconstruction. It is ideal for asset creation, AR/VR, game development, and simulation environments, handling diverse image styles from photos to illustrations. | Upload a single 2D image. Generate a high-quality 3D Gaussian splat representation with controllable density and budget for rendering.
- member workflows:
    - 3d_triposplat_image_to_gaussian_splat
    - image_to_gaussian_splat_triposplat
- node clusters (required structure):
    - model loading: UNETLoader, VAELoader (x2)
    - conditioning: TripoSplatConditioning
    - sampling: KSampler
    - decoding: VAEDecodeTripoSplat
    - output: PreviewImage
    - other operations: CLIPVisionLoader, ComfySwitchNode (x2), InvertMask (x2), JoinImageWithAlpha, LoadBackgroundRemovalModel, RemoveBackground, TripoSplatPreprocessImage, TripoSplatSamplingPreview
- paired/multiple required: InvertMask x2, VAELoader x2
- optional roles: MarkdownNote, SaveGLB, CreateCameraInfo, CreateVideo, LoadImage, RenderSplat, SaveVideo, SplatToFile3D, SplatToMesh
- unresolved nodes: MarkdownNote


# Upscale  (`upscale`)  -  13 workflow(s), 3 model(s)

## Upscale / Generic  (`upscale__generic`)  -  9 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to upscale / enhance an image.
- example request: "build an image workflow"
- description: Local, simple image upscaling via specified ESRGAN model. 1 image -> 1 upscaled image output. Supports various models. | Upload a video to upscale it using a fast GAN model. The output resolution is determined by your chosen model (e.g., 2x or 4x). For best results, upscale first, then resize to your target resolution. | Upload an image and select an interpolation method. Upscale the image using traditional algorithms for faster processing with minimal detail changes. | Upscale and restore video footage using SeedVR2 3B Int8, a one-step diffusion model that enhances resolution while maintaining temporal consistency across frames. This workflow takes one degraded video as input and outputs a single high-resolution, restored video. Ideal for restoring old or degraded footage, upscaling low-resolution videos, and enhancing archival media. | Upscale images using SeedVR2 3B Int8, a one-step diffusion-based video restoration model that produces high-quality results with improved temporal consistency. Ideal for restoring degraded footage, upscaling low-resolution videos, and enhancing archival media. | Upscale images using SeedVR2 7B Int8, a one-step diffusion model that enhances resolution through adversarial training and adaptive window attention. Input 1 image and receive 1 upscaled output with improved detail and clarity. Ideal for restoring low-resolution photos, preparing assets for print, or sharpening lightly degraded digital images. | Upscales video to 4× resolution using a GAN-based upscaling model. | [Local] Image upscale via UltimateSDUpscale. image -> image. Splits the input image into tiles for stable upscaling. | [Local] Upscale via Impact MakeTileSEGS and Detailer. 1 image -> 1 image. Select for tiled processing that prevents memory overload on large images.
- member workflows:
    - ComfyUI-Impact-Pack__4-MakeTileSEGS-Upscale
    - ComfyUI_UltimateSDUpscale__basic-usdu
    - upscale_using_model
    - utility-gan_upscaler
    - utility_interpolation_image_upscale
    - utility_seedvr2_3b_int8_upscale_image
    - utility_seedvr2_3b_int8_upscale_video
    - utility_seedvr2_7b_int8_upscale_image
    - video_upscale_gan_x4
- node clusters (required structure):
    - (none resolved)
- optional roles: Note, CLIPTextEncode, DetailerForEachDebugPipe, ImpactMakeTileSEGS, MarkdownNote, PreviewImage, SEGSPreview, CheckpointLoaderSimple, CreateVideo, EmptyLatentImage, GetVideoComponents, ImageCompare
- unresolved nodes: MarkdownNote, Note, Reroute, workflow>MAKE_BASIC_PIPE

## Upscale / Flux  (`upscale__flux`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to upscale / enhance an image using Flux, Z-Image.
- example request: "build an image workflow using Flux"
- description: Input a latent image from a diffusion model and select a PiD checkpoint. Generate a high-resolution decoded image with 4x or 8x upscaling in a single pass. | Local image upscaling using UltimateSD upscale node (this uses a diffusion model for the upscale process, allowing a creative upscale that invents details). Setup with Flux-1 dev fp8. 1 image -> 1 upscaled image output.
- member workflows:
    - upscale_ultimateSD
    - utility_pid_latent_upscale_dit
- node clusters (required structure):
    - conditioning: CLIPTextEncode (x2)
    - output: SaveImage
- paired/multiple required: CLIPTextEncode x2
- optional roles: CLIPLoader, ConditioningZeroOut, MarkdownNote, UNETLoader, VAEDecode, VAELoader, BasicScheduler, CheckpointLoaderSimple, EmptyChromaRadianceLatentImage, EmptySD3LatentImage, ImageCompare, KSampler
- unresolved nodes: MarkdownNote

## Upscale / Z-Image  (`upscale__z_image`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to upscale / enhance an image using Z-Image.
- example request: "build an image workflow using Z-Image"
- description: Upload an image to upscale it to 2K resolution using the Z-Image-Turbo model. | Upscales images to higher resolution using Z-Image-Turbo.
- member workflows:
    - image_upscale_z_image_turbo
    - utility_z_image_turbo_2k_upscaler.app
- node clusters (required structure):
    - model loading: CLIPLoader, UNETLoader, UpscaleModelLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: ImageScaleBy, ImageScaleToTotalPixels, ImageUpscaleWithModel, ModelSamplingAuraFlow
- paired/multiple required: CLIPTextEncode x2
- optional roles: MarkdownNote, ImageCompare, LoadImage, SaveImage
- unresolved nodes: MarkdownNote


# First / Last Frame to Video  (`first_last_frame_to_video`)  -  8 workflow(s), 4 model(s)

## First / Last Frame to Video / LTX-2  (`first_last_frame_to_video__ltx_2`)  -  4 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using LTX-2.
- example request: "build a video workflow using LTX-2"
- description: Generate a video from first and last frame reference images using LTX-2.5, blending both frames into a single animated sequence. Ideal for creating transition shots, morphing between two subjects or scenes, and building multi-shot narratives from still references. | Generates a video interpolating between first and last keyframes using LTX-2.3. | Generates a video that interpolates between the first and last keyframes using LTX-2.3, including optional audio.
- member workflows:
    - first_last_frame_to_video
    - first_last_frame_to_video_ltx_2_3
    - video_ltx2_3_flf2v
    - video_ltx2_5_flf2v
- node clusters (required structure):
    - conditioning: CLIPTextEncode (x2), LTXVConditioning, ManualSigmas
    - latent / canvas: LTXVEmptyLatentAudio
    - sampling: SamplerCustomAdvanced, SamplerEulerAncestral
    - decoding: LTXVAudioVAEDecode, VAEDecodeTiled
    - other operations: ComfyMathExpression, CreateVideo, EmptyLTXVLatentVideo, GetImageSize, LTXVAddGuide (x2), LTXVConcatAVLatent, LTXVCropGuides, LTXVPreprocess (x2), LTXVSeparateAVLatent, PrimitiveInt (x4), RandomNoise, ResizeImageMaskNode (x2)
- paired/multiple required: CLIPTextEncode x2, LTXVAddGuide x2, LTXVPreprocess x2, ResizeImageMaskNode x2
- optional roles: MarkdownNote, CLIPLoader, LoadImage, VAELoader, CFGGuider, CheckpointLoaderSimple, GetVideoComponents, LTXAVTextEncoderLoader, LTXVAudioVAELoader, LTXVDualCFGGuider, SaveVideo, TextGenerateLTX2Prompt
- unresolved nodes: MarkdownNote

## First / Last Frame to Video / WAN 2.2  (`first_last_frame_to_video__wan_2_2`)  -  2 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Creates a smooth video using 6 key frames.It auto fills in the motion between frames and stitches the segments together seamlessly.
- member workflows:
    - templates-6-key-frames
    - video_wan2_2_14B_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, UNETLoader (x2), VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSamplerAdvanced (x2)
    - decoding: VAEDecode
    - other operations: CreateVideo, ModelSamplingSD3 (x2), WanFirstLastFrameToVideo
- paired/multiple required: CLIPTextEncode x2, KSamplerAdvanced x2, ModelSamplingSD3 x2, UNETLoader x2, LoadImage x2
- optional roles: LoraLoaderModelOnly, Note, BatchImagesNode, GetVideoComponents, MarkdownNote, SaveVideo, VHS_VideoCombine
- unresolved nodes: MarkdownNote, Note

## First / Last Frame to Video / WAN  (`first_last_frame_to_video__wan`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using WAN.
- example request: "build a video workflow using WAN"
- description: Generate videos by controlling first and last frames using Wan 2.1 FLF2V.
- member workflows:
    - wan2.1_flf2v_720_f16
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CLIPVisionEncode (x2), CLIPVisionLoader, CreateVideo, MarkdownNote, ModelSamplingSD3, WanFirstLastFrameToVideo
- paired/multiple required: CLIPTextEncode x2, CLIPVisionEncode x2, LoadImage x2
- unresolved nodes: MarkdownNote

## First / Last Frame to Video / WAN VACE  (`first_last_frame_to_video__wan_vace`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to generate a video interpolating between a first and last frame using WAN VACE.
- example request: "build a video workflow using WAN VACE"
- description: Generate a video interpolating between a first and last frame using WAN VACE. Structurally it loads a diffusion model; uses a VAE; encodes a text prompt; runs a diffusion sampler; decodes the latent to pixels. Boundary inputs: IMAGE, INT; outputs: IMAGE, MASK.
- member workflows:
    - video_wan_vace_flf2v
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, LoraLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage (x2), VHS_VideoCombine
    - other operations: CreateVideo, GetVideoComponents, ImageBatch (x4), ImageToMask, MaskToImage (x2), ModelSamplingSD3, PrimitiveInt (x4), RepeatImageBatch, SolidMask (x2), TrimVideoLatent, WanVaceToVideo
- paired/multiple required: ImageBatch x4, CLIPTextEncode x2, LoadImage x2, MaskToImage x2, PreviewImage x2, SolidMask x2


# Video Tools  (`video_tools`)  -  8 workflow(s), 3 model(s)

## Video Tools / Generic  (`video_tools__generic`)  -  6 workflow(s)  -  source: mixed
- execution: local
- when to use: Use to generate a video.
- example request: "build a video workflow"
- description: Concatenates two videos end-to-end with optional resize, letterbox padding, and audio merge or drop. | Extracts one image frame from a video at a chosen index, with optional trim and FPS control. | Increases video frame rate by synthesizing intermediate frames with a frame interpolation model. | Stitches multiple video clips into a single sequential video file. | [Local] Video matting via MatAnyone. video -> video, image. Pick this when you need background removal using only a first-frame mask. | [Local] video load and combine via VHS_LoadVideoPath and VHS_VideoCombine. Video path and images -> video and preview. Example template from bepic_templates; loads a video, passes images through reformat and size checks, then recombines into video.
- member workflows:
    - ComfyUI_MatAnyone_Kytra__matanyone_with_rembg_anime_for_first_frame
    - bepic_templates__bEpic_defaultTemplate
    - frame_interpolation
    - get_any_video_frame
    - merge_videos
    - video_stitch
- node clusters (required structure):
    - (none resolved)
- optional roles: SetNode, String, bEpicReformat, GetVideoComponents, Image Load, MaskToImage, PreviewImage, VHS_VideoCombine, AudioMerge, BatchImagesNode, CreateVideo, EmptyAudio
- unresolved nodes: Image Load, ImageRemoveBackground+, Note, RemBGSession+, SetNode, String, easy cleanGpuUsed

## Video Tools / Depth Anything  (`video_tools__depth_anything`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to estimate a depth map using Depth Anything.
- example request: "build a video workflow using Depth Anything"
- description: Upload 1 image to generate a depth map for video depth estimation. Output a single depth-processed image ready for use in downstream workflows. Ideal for video generation tasks requiring depth control, 3D scene preprocessing, and animation depth mapping.
- member workflows:
    - utility_depth_anything3_video_depth_estimation
- node clusters (required structure):
    - inputs: LoadVideo
    - output: PreviewImage, SaveVideo
    - other operations: CreateVideo, DA3Inference, DA3Render, GetVideoComponents, LoadDA3Model, MarkdownNote (x2), Video Slice
- paired/multiple required: MarkdownNote x2
- unresolved nodes: MarkdownNote

## Video Tools / SAM3  (`video_tools__sam3`)  -  1 workflow(s)  -  source: custom
- execution: local
- when to use: Use to segment a video using SAM3.
- example request: "build a video workflow using SAM3"
- description: [Local] video segmentation via SAM3. 1 video -> 2 videos. Pick this when you need precise frame-by-frame object tracking using manual point clicks rather than bounding boxes.
- member workflows:
    - ComfyUI-SAM3__video_point_prompt
- node clusters (required structure):
    - inputs: VHS_LoadVideo
    - output: SaveVideo (x2)
    - other operations: CreateVideo (x2), LoadSAM3Model, MaskToImage, SAM3PointCollector, SAM3Propagate, SAM3VideoOutput, SAM3VideoSegmentation
- paired/multiple required: CreateVideo x2, SaveVideo x2


# Inpaint / Outpaint  (`inpaint_outpaint`)  -  7 workflow(s), 3 model(s)

## Inpaint / Outpaint / Flux  (`inpaint_outpaint__flux`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of an image using Flux.
- example request: "build an image workflow using Flux"
- description: Extend images beyond boundaries using Flux.1 outpainting. | Fill missing parts of images using Flux.1 Fill Inpainting. | Inpaints masked image regions using Flux.1 fill [dev], Black Forest Labs' inpainting/outpainting model.
- member workflows:
    - flux_fill_inpaint_example
    - flux_fill_outpaint_example
    - image_inpainting_flux_1_fill_dev
- node clusters (required structure):
    - model loading: DualCLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode, ConditioningZeroOut, FluxGuidance, InpaintModelConditioning
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: DifferentialDiffusion
- optional roles: ImagePadForOutpaint, LoadImage, MarkdownNote, SaveImage
- unresolved nodes: MarkdownNote

## Inpaint / Outpaint / Qwen Image  (`inpaint_outpaint__qwen_image`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of an image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Inpaints masked regions using Qwen-Image, extending its multilingual text rendering to inpainting tasks. | Outpaints beyond image boundaries using Qwen-Image's outpainting capabilities. | Professional inpainting and image editing with Qwen-Image InstantX ControlNet. Supports object replacement, text modification, background changes, and outpainting.
- member workflows:
    - image_inpainting_qwen_image
    - image_outpainting_qwen_image
    - image_qwen_image_instantx_inpainting_controlnet
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2), ControlNetInpaintingAliMamaApply, ControlNetLoader
    - latent / canvas: VAEEncode
    - sampling: KSampler
    - decoding: VAEDecode
    - other operations: GrowMask, ImageBlur, ImageToMask, MaskPreview, MaskToImage, ModelSamplingAuraFlow
- paired/multiple required: CLIPTextEncode x2
- optional roles: MarkdownNote, SaveImage, ImageScaleToMaxDimension, ImageCompositeMasked, LoadImage, Note, FluxKontextImageScale, ImagePadForOutpaint, PreviewImage, SetLatentNoiseMask
- unresolved nodes: MarkdownNote, Note

## Inpaint / Outpaint / Anima  (`inpaint_outpaint__anima`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of an image using Anima.
- example request: "build an image workflow using Anima"
- description: Inpaint anime images with Anima LLLite, a lightweight ControlNet variant that applies low-rank corrections for precise mask-based editing. Input 1 image and receive 1 inpainted output plus a comparison view. Ideal for fixing character details, removing unwanted objects from anime scenes, and seamless background restoration.
- member workflows:
    - image_anima_lllite_image_inpainting
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader, LoraLoaderModelOnly, ModelPatchLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - latent / canvas: EmptyLatentImage
    - sampling: KSampler
    - decoding: VAEDecode
    - output: SaveImage
    - other operations: AnimaLLLiteApply, ComfySwitchNode (x3), ImageCompare, MarkdownNote (x3), MaskPreview, Painter, PrimitiveBoolean, PrimitiveFloat (x2), PrimitiveInt (x2), ResizeImageMaskNode
- paired/multiple required: MarkdownNote x3, CLIPTextEncode x2
- unresolved nodes: MarkdownNote


# Video Inpaint  (`video_inpaint`)  -  7 workflow(s), 3 model(s)

## Video Inpaint / WAN VACE  (`video_inpaint__wan_vace`)  -  3 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint regions of a video using WAN VACE, SAM3.
- example request: "build a video workflow using WAN VACE"
- description: Edit specific regions in videos while preserving surrounding content. Great for object removal or replacement. | Removes objects from video by inpainting masked regions using Wan 2.1 VACE, with SAM3 text-guided segmentation and optional Lightning LoRA turbo mode. | Video Inpaint(Wan2.1 VACE) blueprint
- member workflows:
    - video_inpaint_wan2_1_vace
    - video_inpainting_wan2_1_vace
    - video_wan_vace_inpainting
- node clusters (required structure):
    - model loading: CLIPLoader, LoraLoaderModelOnly, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - sampling: KSampler
    - decoding: VAEDecode
    - output: PreviewImage
    - other operations: CreateVideo, GetImageSize, GetVideoComponents, ImageCompositeMasked, ImageFromBatch, InvertMask, MaskToImage, ModelSamplingSD3, TrimVideoLatent, WanVaceToVideo
- paired/multiple required: CLIPTextEncode x2
- optional roles: MarkdownNote, CheckpointLoaderSimple, GrowMask, ImageToMask, LoadImage, LoadVideo, MaskPreview, RebatchImages, RepeatImageBatch, ResizeImageMaskNode, SAM3_Detect, SaveVideo
- unresolved nodes: MarkdownNote

## Video Inpaint / SAM3  (`video_inpaint__sam3`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint regions of a video using SAM3.
- example request: "build a video workflow using SAM3"
- description: Removes objects from video by inpainting masked regions using VOID (CogVideoX), with SAM3 text-guided segmentation and optional two-pass optical-flow refinement. | Upload a video and mask the object you want to remove. Generate a clean video with the object and its physical interactions deleted.
- member workflows:
    - utility_void_video_inpainting
    - video_inpaint_void
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple, UNETLoader (x2), VAELoader
    - conditioning: BasicScheduler (x2), CFGGuider (x2), CLIPTextEncode (x3), VOIDInpaintConditioning
    - sampling: SamplerCustomAdvanced (x2), VOIDSampler (x2)
    - decoding: VAEDecode (x2)
    - other operations: ComfyMathExpression (x2), ComfySwitchNode, CreateVideo (x2), GetImageSize, GetVideoComponents, ImageFromBatch, MaskPreview, OpticalFlowLoader, PrimitiveBoolean, PrimitiveInt (x4), RandomNoise, SAM3_Detect, TrimAudioDuration, VOIDWarpedNoise, VOIDWarpedNoiseSource
- paired/multiple required: CLIPTextEncode x3, BasicScheduler x2, CFGGuider x2, CreateVideo x2, SamplerCustomAdvanced x2, UNETLoader x2, VAEDecode x2, VOIDSampler x2
- optional roles: MarkdownNote, SaveVideo, LoadVideo
- unresolved nodes: MarkdownNote

## Video Inpaint / WAN 2.2  (`video_inpaint__wan_2_2`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to inpaint masked regions of a video using WAN 2.2.
- example request: "build a video workflow using WAN 2.2"
- description: Efficient video inpainting from start and end frames. 5B model delivers quick iterations for testing workflows. | Generate videos from start and end frames using Wan 2.2 Fun Inpaint.
- member workflows:
    - video_wan2_2_14B_fun_inpaint
    - video_wan2_2_5B_fun_inpaint
- node clusters (required structure):
    - inputs: LoadImage (x2)
    - model loading: CLIPLoader, UNETLoader, VAELoader
    - conditioning: CLIPTextEncode (x2)
    - decoding: VAEDecode
    - output: SaveVideo
    - other operations: CreateVideo, MarkdownNote, ModelSamplingSD3, WanFunInpaintToVideo
- paired/multiple required: CLIPTextEncode x2, LoadImage x2
- optional roles: KSamplerAdvanced, LoraLoaderModelOnly, Note, KSampler
- unresolved nodes: MarkdownNote, Note


# API / Partner Nodes - Character  (`api_partner_nodes_character`)  -  5 workflow(s), 2 model(s)

## API / Partner Nodes - Character / Generic  (`api_partner_nodes_character__generic`)  -  3 workflow(s)  -  source: mixed
- execution: api (API nodes: GeminiImage2Node, OpenAIGPTImageNodeV2)
- when to use: Use to generate a multi-pose character sheet.
- example request: "build an image workflow"
- description: Upload a portrait of your character and create multiple cinematic lighting conditions while maintaining character consistency. | Upload your character and generate 360 turnaround views for full body and close ups. | [API] Character reference sheet via OpenAI GPT Image 2.5, driven by a reference photo. LoadImage (VHS, absolute path) -> OpenAIGPTImageNodeV2 model.images.image_1 + prompt -> bEpicSendToViewer. Prompt arrives on the agentY text/hook wire; filename_prefix from the hook. Wrapped in the agentY hook pipeline (18 extract characters -> 8 sweep prompts -> 135 human review, loop break 133) so it can be re-run per round.
- member workflows:
    - morning_people_character_sheet
    - template_character_portrait_relighting
    - templates-character_sheet
- node clusters (required structure):
    - (none resolved)
- optional roles: GeminiImage2Node, SaveImage, AgentYImageCollector, ImageStitch, LoadImage, OpenAIGPTImageNodeV2, bEpicSendToViewer
- unresolved nodes: Reroute

## API / Partner Nodes - Character / Nano-Banana  (`api_partner_nodes_character__nano_banana`)  -  2 workflow(s)  -  source: custom
- execution: api (API nodes: GeminiImage2Node, GeminiNode)
- when to use: Use to generate a multi-pose character sheet using Nano-Banana.
- example request: "build an image workflow using Nano-Banana"
- description: API character sheet generation FOR FACE CLOSEUPS via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying facial expressions in a single sheet. | API character sheet generation via Nano-Banana Pro. 1 character image -> 1 image output (3x3 sheet). Uses an LLM call to generate a prompt from the reference, then renders 9 character views with varying body pose in a single sheet.
- member workflows:
    - NanoBananaPro_3x3CharacterSheet
    - NanoBananaPro_3x3CharacterSheet_closeups
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: GeminiImage2Node, GeminiNode, PrimitiveStringMultiline


# Text Tools  (`text_tools`)  -  5 workflow(s), 2 model(s)

## Text Tools / Gemini  (`text_tools__gemini`)  -  3 workflow(s)  -  source: mixed
- execution: hybrid (API nodes: GeminiNode)
- when to use: Use to caption an image as text using Gemini.
- example request: "build a text workflow using Gemini"
- description: Generates descriptive captions for images using Google's Gemini multimodal LLM. | Generates descriptive captions for video input using Google's Gemini multimodal LLM. | [API] motion prompt generation via Gemini, analyses a video and output a desscription of the motion in it. 1 video input -> 1 output. Generates descriptive motion prompts for video generation.
- member workflows:
    - image_captioning_gemini
    - video_captioning_gemini
    - video_gemini_motionPromptGeneration
- node clusters (required structure):
    - other operations: GeminiNode
- optional roles: CreateVideo, GetVideoComponents, ImageResizeKJv2, LoadVideo, easy saveText
- unresolved nodes: easy saveText

## Text Tools / Generic  (`text_tools__generic`)  -  2 workflow(s)  -  source: official
- execution: hybrid (API nodes: GeminiNode)
- when to use: Use to expand a short prompt into a detailed one.
- example request: "build an image workflow"
- description: Expands short text prompts into detailed descriptions using a text generation model for better generation quality. | Selects one line from multiline text by zero-based index for batch or list-driven prompt workflows.
- member workflows:
    - prompt_enhance
    - select_per_line_text_by_index
- node clusters (required structure):
    - (none resolved)
- optional roles: GeminiNode, RegexExtract


# API / Partner Nodes - Inpaint / Outpaint  (`api_partner_nodes_inpaint_outpaint`)  -  3 workflow(s), 2 model(s)

## API / Partner Nodes - Inpaint / Outpaint / Generic  (`api_partner_nodes_inpaint_outpaint__generic`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: BriaImageEditNode, OpenAIGPTImageNodeV2)
- when to use: Use to inpaint masked regions of an image.
- example request: "build an image workflow"
- description: A professional Bria-powered outpainting template designed for seamless image extension and edge-aware scene expansion | Edit images using inpainting with OpenAI GPT Image 1 API.
- member workflows:
    - api_bria_image_outpainting
    - api_openai_image_1_inpaint
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
- optional roles: BriaImageEditNode, ImageCompare, ImagePadForOutpaint, MarkdownNote, OpenAIGPTImageNodeV2, PreviewImage
- unresolved nodes: MarkdownNote

## API / Partner Nodes - Inpaint / Outpaint / Nano-Banana  (`api_partner_nodes_inpaint_outpaint__nano_banana`)  -  1 workflow(s)  -  source: custom
- execution: api (API nodes: GeminiNanoBanana2)
- when to use: Use to outpaint / extend an image beyond its borders using Nano-Banana.
- example request: "build an image workflow using Nano-Banana"
- description: API upscale and outpaint via Nano-Banana 2. 1 image -> 1 image output. Upscales the input image while also generating new content around the edges to expand the overall dimensions, guided by the original image's style and content.
- member workflows:
    - NanoBanana2_outpaintUpscale
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: GeminiNanoBanana2


# Character  (`character`)  -  3 workflow(s), 2 model(s)

## Character / SCAIL  (`character__scail`)  -  2 workflow(s)  -  source: official
- execution: local
- when to use: Use to replace a character in a video using SCAIL.
- example request: "build a video workflow using SCAIL"
- description: Replaces a character in a video with a reference image using the SCAIL-2 model for end-to-end controlled animation without intermediate pose maps. Key inputs include a source video, a reference character image, and optional text prompts for style or context. Suitable for animated or live-action footage, multi-character scenes, and creative video editing where direct pose-free animation is needed; works best with moderate-length videos.
- member workflows:
    - character_replacement_scail_2_base
    - character_replacement_scail_2_extend
- node clusters (required structure):
    - model loading: CLIPLoader, CheckpointLoaderSimple, LoraLoaderModelOnly (x2), UNETLoader, VAELoader
    - conditioning: BasicScheduler, CLIPTextEncode (x4)
    - sampling: KSamplerSelect, SamplerCustom
    - decoding: VAEDecode
    - output: PreviewImage (x2)
    - other operations: CLIPVisionEncode, CLIPVisionLoader, ComfyMathExpression (x3), ComfySwitchNode (x3), GetImageSize, GetVideoComponents, ImageFromBatch, ModelSamplingSD3, PrimitiveBoolean (x2), PrimitiveFloat (x2), PrimitiveInt (x5), ResizeImageMaskNode, SAM3_VideoTrack (x2), SCAIL2ColoredMask, WanSCAILToVideo
- paired/multiple required: CLIPTextEncode x4, LoraLoaderModelOnly x2, PreviewImage x2, SAM3_VideoTrack x2
- optional roles: ColorTransfer

## Character / Qwen Image  (`character__qwen_image`)  -  1 workflow(s)  -  source: official
- execution: local
- when to use: Use to generate an image using Qwen Image.
- example request: "build an image workflow using Qwen Image"
- description: Upload an image of your character and get multiple views of that image from different angles
- member workflows:
    - templates-1_click_multiple_character_angles-v1.0
- node clusters (required structure):
    - inputs: LoadImage
    - model loading: CLIPLoader (x8), LoraLoaderModelOnly (x16), UNETLoader (x8), VAELoader (x8)
    - conditioning: TextEncodeQwenImageEditPlus (x16)
    - latent / canvas: VAEEncode (x8)
    - sampling: KSampler (x8)
    - decoding: VAEDecode (x8)
    - output: SaveImage (x8)
    - other operations: CFGNorm (x8), ComfySwitchNode (x24), FluxKontextImageScale (x8), FluxKontextMultiReferenceLatentMethod (x16), MarkdownNote, ModelSamplingAuraFlow (x8), Note (x8), PrimitiveBoolean (x8), PrimitiveFloat (x16), PrimitiveInt (x16)
- paired/multiple required: FluxKontextMultiReferenceLatentMethod x16, LoraLoaderModelOnly x16, TextEncodeQwenImageEditPlus x16, CFGNorm x8, CLIPLoader x8, FluxKontextImageScale x8, KSampler x8, ModelSamplingAuraFlow x8, Note x8, SaveImage x8, UNETLoader x8, VAEDecode x8, VAEEncode x8, VAELoader x8
- unresolved nodes: MarkdownNote, Note


# API / Partner Nodes - Image Edit with ControlNet  (`api_partner_nodes_image_edit_with_controlnet`)  -  2 workflow(s), 1 model(s)

## API / Partner Nodes - Image Edit with ControlNet / Generic  (`api_partner_nodes_image_edit_with_controlnet__generic`)  -  2 workflow(s)  -  source: official
- execution: api (API nodes: RecraftColorRGB, RecraftControls, RecraftStyleV3DigitalIllustration, RecraftStyleV3LogoRaster, RecraftStyleV3RealisticImage, RecraftTextToImageNode, RecraftVectorizeImageNode)
- when to use: Use to generate an image guided by a control map (canny/depth/pose).
- example request: "build an image workflow"
- description: Control style with visual examples, align positioning, and fine-tune objects. Store and share styles for perfect brand consistency. | Generate images with custom color palettes and brand-specific visuals using Recraft.
- member workflows:
    - api_recraft_image_gen_with_color_control
    - api_recraft_image_gen_with_style_control
- node clusters (required structure):
    - output: SaveImage
    - other operations: MarkdownNote (x2), RecraftColorRGB (x3), RecraftControls, RecraftStyleV3DigitalIllustration, RecraftStyleV3LogoRaster, RecraftStyleV3RealisticImage, RecraftTextToImageNode, RecraftVectorizeImageNode, SaveSVGNode
- paired/multiple required: RecraftColorRGB x3, MarkdownNote x2
- unresolved nodes: MarkdownNote


# API / Partner Nodes - Preprocessors / Estimation  (`api_partner_nodes_preprocessors_estimation`)  -  1 workflow(s), 1 model(s)

## API / Partner Nodes - Preprocessors / Estimation / Generic  (`api_partner_nodes_preprocessors_estimation__generic`)  -  1 workflow(s)  -  source: official
- execution: api (API nodes: BriaGuidedRemoveBackground)
- when to use: Use to remove the background from an image.
- example request: "build an image workflow"
- description: Remove the background from an image with Bria while keeping whatever a text prompt describes, so one subject can be cut out of a busy scene instead of losing everything that is not background. The original pixels come back at full resolution with a new alpha channel. Ideal for product cutouts, portrait isolation and compositing prep.
- member workflows:
    - api_bria_guided_remove_background
- node clusters (required structure):
    - inputs: LoadImage
    - output: SaveImage
    - other operations: BriaGuidedRemoveBackground, ImageCompare, MarkdownNote
- unresolved nodes: MarkdownNote

